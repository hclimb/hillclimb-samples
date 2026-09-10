"""End-to-end coverage for the alternative target-gradient signal pipeline.

The loss math itself lives in ``test_target_gradient_signals``. These tests
cover everything around it: answer-span extraction, candidate artifacts, the
domain verifiers, the dataset builders, and the trainer's microbatch slicing.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import torch

from SFT.data.target_candidates import (
    Candidate,
    CandidateGroup,
    answer_span_is_degenerate,
    candidates_path,
    final_answer_char_span,
    load_candidate_groups,
    write_candidate_groups,
)
from SFT.data.target_signal_dataset import (
    POSITIVE_SOURCE_GENERATED,
    build_target_signal_features,
    restrict_labels_to_final_answer,
)
from SFT.data.target_verifiers import (
    MbppVerifier,
    extract_boxed_answer,
    recover_if_constraints,
)
from SFT.train.target_signal import (
    ANSWER_ONLY_CE,
    CORRECT_INCORRECT_MARGIN,
    GROUP_IDS_KEY,
    NLL,
    REWARD_WEIGHTED_SFT,
    REWARDS_KEY,
    GroupedTargetCollator,
    causal_token_mean_nll,
    compute_target_signal_loss_from_batch,
)
from SFT.train.trainer import LayerWiseSubsetTrainer

# Negatives must clear MIN_INFORMATIVE_NEGATIVE_CHARS (40) to be selectable, so
# fixtures use text long enough to represent a real solution attempt.
_WELL_FORMED_NEGATIVE = "A complete but incorrect derivation ending in \\boxed{99}."
_MALFORMED_NEGATIVE = "Sorry, I am not able to answer that question at all here."

_IM_START = 1
_IM_END = 2
_SPECIALS = {"<|im_start|>": _IM_START, "<|im_end|>": _IM_END}
_INVERSE_SPECIALS = {value: key for key, value in _SPECIALS.items()}


class CharTokenizerStub:
    """One token per character, plus the two Qwen chat delimiters.

    A bijection between characters and tokens makes the character-offset to
    token-index mapping in ``restrict_labels_to_final_answer`` exactly checkable
    rather than approximately checkable.
    """

    chat_template = "<|im_start|>...<|im_end|>"
    unk_token_id = -1
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return _SPECIALS.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        result = []
        index = 0
        while index < len(text):
            for marker, token_id in _SPECIALS.items():
                if text.startswith(marker, index):
                    result.append(token_id)
                    index += len(marker)
                    break
            else:
                result.append(ord(text[index]) + 100)
                index += 1
        return result

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=None):
        del clean_up_tokenization_spaces
        pieces = []
        for token_id in ids:
            if int(token_id) in _INVERSE_SPECIALS:
                if not skip_special_tokens:
                    pieces.append(_INVERSE_SPECIALS[int(token_id)])
                continue
            pieces.append(chr(int(token_id) - 100))
        return "".join(pieces)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kwargs
    ):
        del tools, kwargs
        rendered = ""
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n{message.get('content', '')}"
            rendered += "<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return self.encode(rendered) if tokenize else rendered


def _row(row_id: str, user: str, assistant: str, **metadata) -> dict:
    return {
        "id": row_id,
        "prompt_hash": f"hash-{row_id}",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata,
    }


class AnswerSpanTests(unittest.TestCase):
    def test_math_span_starts_at_the_last_boxed_answer(self):
        text = "First \\boxed{1} was wrong. The answer is \\boxed{42}.\n\n"
        start, end = final_answer_char_span("math", text)
        self.assertEqual(text[start:end], "\\boxed{42}.")

    def test_mbpp_span_is_the_last_fenced_block(self):
        text = "Explanation.\n```python\nx = 1\n```\nMore prose.\n```python\ndef f():\n    return 2\n```\n"
        start, end = final_answer_char_span("mbpp", text)
        self.assertEqual(text[start:end], "```python\ndef f():\n    return 2\n```")

    def test_missing_marker_falls_back_to_the_whole_turn(self):
        text = "No boxed answer here.   \n"
        self.assertEqual(final_answer_char_span("math", text), (0, len(text.rstrip())))

    def test_instruction_following_has_no_separable_answer(self):
        self.assertTrue(answer_span_is_degenerate("precise_if"))
        text = "A response with several paragraphs."
        self.assertEqual(final_answer_char_span("precise_if", text), (0, len(text)))

    def test_unknown_target_is_rejected_rather_than_guessed(self):
        with self.assertRaisesRegex(KeyError, "no final-answer rule"):
            final_answer_char_span("not_a_target", "text")

    def test_empty_assistant_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            final_answer_char_span("math", "   \n")


class LabelRestrictionTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizerStub()

    def _encode(self, target, assistant):
        rows = [_row("r0", "question", assistant)]
        features, stats = build_target_signal_features(
            ANSWER_ONLY_CE, rows, self.tokenizer, 4096, target=target
        )
        candidate = features[0]["candidates"][0]
        mask = candidate["labels"] != -100
        return self.tokenizer.decode(
            candidate["input_ids"][mask], skip_special_tokens=False
        ), stats

    def test_only_the_boxed_answer_is_supervised(self):
        supervised, stats = self._encode(
            "math", "Long derivation goes here. Therefore \\boxed{7}."
        )
        self.assertEqual(supervised, "\\boxed{7}.<|im_end|>")
        self.assertLess(stats["answer_token_fraction"], 0.5)

    def test_answer_only_labels_are_a_subset_of_the_reference_labels(self):
        rows = [_row("r0", "question", "Reasoning first. \\boxed{3}")]
        full, _ = build_target_signal_features(
            NLL, rows, self.tokenizer, 4096, target="math"
        )
        answer, _ = build_target_signal_features(
            ANSWER_ONLY_CE, rows, self.tokenizer, 4096, target="math"
        )
        full_labels = full[0]["candidates"][0]["labels"]
        answer_labels = answer[0]["candidates"][0]["labels"]
        supervised = answer_labels != -100
        self.assertTrue(bool(supervised.any()))
        self.assertTrue(torch.equal(answer_labels[supervised], full_labels[supervised]))
        self.assertTrue(bool(((answer_labels == -100) | (full_labels != -100)).all()))

    def test_degenerate_domain_keeps_every_reference_label(self):
        rows = [_row("r0", "instruction", "A response.")]
        full, _ = build_target_signal_features(
            NLL, rows, self.tokenizer, 4096, target="precise_if"
        )
        answer, stats = build_target_signal_features(
            ANSWER_ONLY_CE, rows, self.tokenizer, 4096, target="precise_if"
        )
        self.assertTrue(stats["answer_span_degenerate"])
        self.assertTrue(
            torch.equal(
                full[0]["candidates"][0]["labels"], answer[0]["candidates"][0]["labels"]
            )
        )

    def test_restriction_never_masks_every_label(self):
        rows = [_row("r0", "q", "\\boxed{}")]
        features, _ = build_target_signal_features(
            NLL, rows, self.tokenizer, 4096, target="math"
        )
        encoded = features[0]["candidates"][0]
        restricted = restrict_labels_to_final_answer(encoded, self.tokenizer, "math")
        self.assertTrue(bool((restricted != -100).any()))


class VerifierTests(unittest.TestCase):
    def test_boxed_extraction_is_brace_balanced_and_takes_the_last(self):
        self.assertEqual(extract_boxed_answer("\\boxed{1} then \\boxed{\\frac{a}{b}}"), "\\frac{a}{b}")
        self.assertIsNone(extract_boxed_answer("no answer"))
        self.assertIsNone(extract_boxed_answer("\\boxed{unclosed"))

    def test_ifeval_constraints_are_recovered_from_their_own_templates(self):
        prompt = (
            "Write about trains. Include keywords ['edge', 'gas'] in the response. "
            "There should be 6 paragraphs. Paragraphs and only paragraphs are separated "
            "with each other by two new lines as if it was '\\n\\n' in python. "
            "Paragraph 4 must start with word board. "
            "Wrap your entire response with double quotation marks."
        )
        recovered = dict(recover_if_constraints(prompt))
        self.assertEqual(
            recovered["keywords:existence"], {"keywords": ["edge", "gas"]}
        )
        self.assertEqual(
            recovered["length_constraints:nth_paragraph_first_word"],
            {"num_paragraphs": 6, "nth_paragraph": 4, "first_word": "board"},
        )
        self.assertIn("startend:quotation", recovered)
        # The nth-paragraph form must not also register the divider form.
        self.assertNotIn("length_constraints:number_paragraphs", recovered)

    def test_unconstrained_prompt_recovers_nothing(self):
        self.assertEqual(recover_if_constraints("Just answer the question."), [])

    def test_mbpp_verifier_runs_tests_and_separates_failure_modes(self):
        verifier = MbppVerifier(timeout_sec=30.0)
        row = _row(
            "mbpp_train::1",
            "Write add(a, b).",
            "```python\ndef add(a, b):\n    return a + b\n```",
            test_list=["assert add(1, 2) == 3"],
            test_setup_code="",
        )
        context = verifier.prepare(row)
        self.assertIsNotNone(context)
        self.assertTrue(verifier.verify(context, row["messages"][1]["content"]).correct)

        wrong = verifier.verify(context, "```python\ndef add(a, b):\n    return a - b\n```")
        self.assertFalse(wrong.correct)
        self.assertEqual(wrong.status, "tests_failed")

        broken = verifier.verify(context, "```python\ndef add(a, b\n```")
        self.assertEqual(broken.status, "syntax_error")

    def test_row_without_tests_is_not_verifiable(self):
        self.assertIsNone(MbppVerifier().prepare(_row("m", "task", "code")))


class CandidateArtifactTests(unittest.TestCase):
    def test_round_trip_and_negative_ranking(self):
        import tempfile

        group = CandidateGroup(
            id="row-0",
            prompt_hash="hash",
            verifiable=True,
            candidates=(
                Candidate("gold", "correct", 1.0, "reference", {"status": "reference"}),
                Candidate(_MALFORMED_NEGATIVE, "incorrect", 0.0, "generated", {"status": "prediction_parse_error"}),
                Candidate(_WELL_FORMED_NEGATIVE, "incorrect", 0.0, "generated", {"status": "incorrect"}),
            ),
        )
        with tempfile.TemporaryDirectory() as workdir:
            path = candidates_path(workdir, "b" * 64, "math")
            write_candidate_groups(path, [group])
            loaded = load_candidate_groups(path)
        restored = loaded["row-0"]
        self.assertEqual(restored.to_json(), group.to_json())
        # A well-formed wrong answer must outrank an unparseable one.
        self.assertEqual(restored.ranked_negatives()[0].content, _WELL_FORMED_NEGATIVE)

    def test_collapsed_negatives_are_dropped_not_merely_deranked(self):
        # High-temperature sampling emits things like a lone replacement char.
        # Such a "negative" must not become the pair the margin loss trains on.
        group = CandidateGroup(
            "row-1",
            "hash",
            True,
            (
                Candidate("gold", "correct", 1.0, "reference", {"status": "reference"}),
                Candidate("�", "incorrect", 0.0, "generated", {"status": "syntax_error"}),
                Candidate("  ", "incorrect", 0.0, "generated", {"status": "incorrect"}),
            ),
        )
        self.assertEqual(group.ranked_negatives(), ())
        # The artifact still holds them; only the training-time selection drops them.
        self.assertEqual(len(group.with_roles("incorrect")), 2)
        self.assertEqual(len(group.ranked_negatives(min_chars=1)), 1)

    def test_duplicate_ids_are_rejected(self):
        import tempfile
        from pathlib import Path

        group = CandidateGroup("dup", "h", True, (Candidate("x", "correct", 1.0, "reference"),))
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "candidates.jsonl"
            payload = json.dumps(group.to_json())
            path.write_text(payload + "\n" + payload + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate candidate group id"):
                load_candidate_groups(path)


def _candidate_groups():
    return {
        "row-0": CandidateGroup(
            "row-0",
            "h0",
            True,
            (
                Candidate("gold zero \\boxed{1}", "correct", 1.0, "reference", {"status": "reference"}),
                Candidate("model wrong: a full derivation ending \\boxed{9}", "incorrect", 0.0, "generated", {"status": "incorrect"}),
                Candidate("model right \\boxed{1}", "correct", 1.0, "generated", {"status": "correct"}),
            ),
        ),
        "row-1": CandidateGroup(
            "row-1",
            "h1",
            True,
            (
                Candidate("gold one \\boxed{2}", "correct", 1.0, "reference", {"status": "reference"}),
                Candidate("also wrong: a full derivation ending \\boxed{8}", "incorrect", 0.0, "generated", {"status": "incorrect"}),
            ),
        ),
        # No negative: usable by reward weighting, not by the margin objective.
        "row-2": CandidateGroup(
            "row-2",
            "h2",
            True,
            (Candidate("gold two \\boxed{3}", "correct", 1.0, "reference", {"status": "reference"}),),
        ),
    }


class GroupedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizerStub()
        self.rows = [
            _row("row-0", "q0", "gold zero \\boxed{1}"),
            _row("row-1", "q1", "gold one \\boxed{2}"),
            _row("row-2", "q2", "gold two \\boxed{3}"),
        ]
        self.groups = _candidate_groups()

    def test_margin_pairs_the_reference_with_a_verified_negative(self):
        features, stats = build_target_signal_features(
            CORRECT_INCORRECT_MARGIN,
            self.rows,
            self.tokenizer,
            4096,
            target="math",
            candidate_groups=self.groups,
        )
        self.assertEqual(stats["n_prompts"], 2)
        self.assertEqual(stats["skipped"]["no_negative"], 1)
        self.assertEqual(stats["malformed_negative_pairs"], 0)
        roles = [candidate["role"] for candidate in features[0]["candidates"]]
        self.assertEqual(roles, ["correct", "incorrect"])
        chosen = features[0]["candidates"][0]
        supervised = chosen["labels"] != -100
        self.assertIn(
            "gold zero",
            self.tokenizer.decode(chosen["input_ids"][supervised]),
        )

    def test_margin_can_use_a_generated_positive_instead(self):
        features, stats = build_target_signal_features(
            CORRECT_INCORRECT_MARGIN,
            self.rows,
            self.tokenizer,
            4096,
            target="math",
            candidate_groups=self.groups,
            positive_source=POSITIVE_SOURCE_GENERATED,
        )
        # Only row-0 has a generated correct trajectory.
        self.assertEqual(stats["n_prompts"], 1)
        self.assertEqual(stats["skipped"]["no_positive"], 1)
        chosen = features[0]["candidates"][0]
        supervised = chosen["labels"] != -100
        self.assertIn(
            "model right", self.tokenizer.decode(chosen["input_ids"][supervised])
        )

    def test_margin_without_any_negative_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "no correct/incorrect pairs survived"):
            build_target_signal_features(
                CORRECT_INCORRECT_MARGIN,
                self.rows[2:],
                self.tokenizer,
                4096,
                target="math",
                candidate_groups=self.groups,
            )

    def test_reward_weighting_keeps_every_prompt_and_drops_zero_weight_rows(self):
        features, stats = build_target_signal_features(
            REWARD_WEIGHTED_SFT,
            self.rows,
            self.tokenizer,
            4096,
            target="math",
            candidate_groups=self.groups,
        )
        self.assertEqual(stats["n_prompts"], 3)
        # Default incorrect_reward=0 keeps gold everywhere plus row-0's correct
        # generation: rejection-sampling fine-tuning.
        self.assertEqual(stats["n_trajectories"], 4)
        self.assertEqual(stats["n_weighted_negative_trajectories"], 0)

    def test_positive_incorrect_reward_keeps_wrong_trajectories_downweighted(self):
        features, stats = build_target_signal_features(
            REWARD_WEIGHTED_SFT,
            self.rows,
            self.tokenizer,
            4096,
            target="math",
            candidate_groups=self.groups,
            incorrect_reward=0.25,
        )
        self.assertEqual(stats["n_trajectories"], 6)
        self.assertEqual(stats["n_weighted_negative_trajectories"], 2)
        rewards = [candidate["reward"] for candidate in features[0]["candidates"]]
        self.assertEqual(sorted(rewards), [0.25, 1.0, 1.0])

    def test_candidate_cap_keeps_the_reference_first(self):
        _, stats = build_target_signal_features(
            REWARD_WEIGHTED_SFT,
            self.rows,
            self.tokenizer,
            4096,
            target="math",
            candidate_groups=self.groups,
            max_candidates_per_prompt=1,
        )
        self.assertEqual(stats["n_trajectories"], 3)
        self.assertEqual(stats["n_positive_trajectories"], 3)

    def test_grouped_signals_require_candidates(self):
        with self.assertRaisesRegex(ValueError, "requires pre-generated candidates"):
            build_target_signal_features(
                CORRECT_INCORRECT_MARGIN, self.rows, self.tokenizer, 4096, target="math"
            )

    def test_features_collate_into_a_grouped_batch(self):
        features, _ = build_target_signal_features(
            CORRECT_INCORRECT_MARGIN,
            self.rows,
            self.tokenizer,
            4096,
            target="math",
            candidate_groups=self.groups,
        )
        batch = GroupedTargetCollator(pad_token_id=0)(features)
        self.assertEqual(batch[GROUP_IDS_KEY].tolist(), [0, 0, 1, 1])
        self.assertEqual(batch["roles"].tolist(), [1, -1, 1, -1])
        self.assertEqual(batch["input_ids"].shape[0], 4)


class TopUpMergeTests(unittest.TestCase):
    """Second-pass sampling must add negatives without disturbing the first pass."""

    def setUp(self):
        from SFT.data.build_target_candidates import ids_missing_negatives, merge_top_up

        self.ids_missing_negatives = ids_missing_negatives
        self.merge_top_up = merge_top_up
        self.existing = {
            "has-negative": CandidateGroup(
                "has-negative",
                "h0",
                True,
                (
                    Candidate("gold", "correct", 1.0, "reference", {"status": "reference"}),
                    Candidate("a wrong answer long enough to count as an attempt", "incorrect", 0.0, "generated", {"status": "incorrect"}),
                ),
            ),
            "all-correct": CandidateGroup(
                "all-correct",
                "h1",
                True,
                (
                    Candidate("gold b", "correct", 1.0, "reference", {"status": "reference"}),
                    Candidate("right", "correct", 1.0, "generated", {"status": "correct"}),
                ),
            ),
            "unverifiable": CandidateGroup(
                "unverifiable",
                "h2",
                False,
                (Candidate("gold c", "correct", 1.0, "reference", {"status": "reference"}),),
            ),
        }

    def test_only_verifiable_prompts_without_a_negative_are_selected(self):
        # The unverifiable prompt is excluded: more sampling cannot score it.
        self.assertEqual(self.ids_missing_negatives(self.existing), ["all-correct"])

    def test_merge_adds_negatives_and_leaves_the_reference_alone(self):
        fresh = [
            CandidateGroup(
                "all-correct",
                "h1",
                True,
                (
                    Candidate("gold b", "correct", 1.0, "reference", {"status": "reference"}),
                    Candidate("newly wrong, and long enough to count as an attempt", "incorrect", 0.0, "generated", {"status": "incorrect"}),
                ),
            )
        ]
        merged, counts = self.merge_top_up(self.existing, fresh)
        self.assertEqual(counts["new_negatives"], 1)
        self.assertEqual(counts["topped_up_prompts"], 1)
        group = merged["all-correct"]
        # Exactly one reference survives, and it is the original.
        references = [c for c in group.candidates if c.origin == "reference"]
        self.assertEqual([c.content for c in references], ["gold b"])
        self.assertEqual(group.ranked_negatives()[0].content, "newly wrong, and long enough to count as an attempt")
        # Untouched groups are preserved by identity, not rebuilt.
        self.assertIs(merged["has-negative"], self.existing["has-negative"])

    def test_repeated_text_is_not_duplicated(self):
        fresh = [
            CandidateGroup(
                "all-correct",
                "h1",
                True,
                (
                    Candidate("right", "correct", 1.0, "generated", {"status": "correct"}),
                    Candidate("right", "correct", 1.0, "generated", {"status": "correct"}),
                ),
            )
        ]
        merged, counts = self.merge_top_up(self.existing, fresh)
        self.assertEqual(counts["added"], 0)
        self.assertEqual(counts["duplicate_dropped"], 2)
        self.assertEqual(len(merged["all-correct"].candidates), 2)

    def test_unknown_prompt_id_is_rejected(self):
        fresh = [CandidateGroup("ghost", "h", True, ())]
        with self.assertRaisesRegex(KeyError, "unknown prompt id"):
            self.merge_top_up(self.existing, fresh)


class _FakeTrainer:
    """Just enough of the trainer to exercise its slicing logic."""

    _target_logical_slices = LayerWiseSubsetTrainer._target_logical_slices

    def __init__(self, mode, **args):
        self.target_signal_mode = mode
        defaults = {"target_microbatch_size": None, "target_signal_groups_per_microbatch": None}
        self.args = SimpleNamespace(**{**defaults, **args})


class TrainerSlicingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.labels = torch.tensor(
            [
                [-100, 1, 2, 3, 4],
                [-100, 2, 3, -100, -100],
                [-100, 4, 5, 6, 7],
                [-100, 5, -100, -100, -100],
            ]
        )
        self.logits = torch.randn(4, 5, 9, dtype=torch.double)
        self.batch = {
            "input_ids": torch.ones(4, 5, dtype=torch.long),
            "labels": self.labels,
            GROUP_IDS_KEY: torch.tensor([0, 0, 1, 1]),
            "roles": torch.tensor([1, -1, 1, -1]),
            REWARDS_KEY: torch.tensor([1.0, 0.25, 1.0, 0.5]),
        }

    def _slices(self, mode, **args):
        return list(_FakeTrainer(mode, **args)._target_logical_slices(self.batch, self.labels))

    def test_cross_entropy_slices_reproduce_the_original_token_fractions(self):
        tokens = (self.labels[:, 1:] != -100).sum(dim=1)
        total = tokens.sum()
        for mode in (NLL, ANSWER_ONLY_CE):
            slices = self._slices(mode, target_microbatch_size=2)
            self.assertEqual([(start, end) for start, end, _ in slices], [(0, 2), (2, 4)])
            for start, end, fraction in slices:
                expected = tokens[start:end].sum().double() / total
                self.assertAlmostEqual(float(fraction), float(expected), places=12)

    def test_reward_weighted_slices_recompose_the_unsliced_loss(self):
        full = compute_target_signal_loss_from_batch(
            self.logits, self.batch, mode=REWARD_WEIGHTED_SFT
        )
        pieces = []
        for start, end, fraction in self._slices(
            REWARD_WEIGHTED_SFT, target_microbatch_size=1
        ):
            chunk = {key: value[start:end] for key, value in self.batch.items()}
            pieces.append(
                compute_target_signal_loss_from_batch(
                    self.logits[start:end], chunk, mode=REWARD_WEIGHTED_SFT
                )
                * fraction
            )
        torch.testing.assert_close(torch.stack(pieces).sum(), full)

    def test_margin_slices_on_group_boundaries_and_recomposes(self):
        full = compute_target_signal_loss_from_batch(
            self.logits, self.batch, mode=CORRECT_INCORRECT_MARGIN
        )
        slices = self._slices(
            CORRECT_INCORRECT_MARGIN, target_signal_groups_per_microbatch=1
        )
        self.assertEqual([(start, end) for start, end, _ in slices], [(0, 2), (2, 4)])
        pieces = []
        for start, end, fraction in slices:
            chunk = {key: value[start:end] for key, value in self.batch.items()}
            pieces.append(
                compute_target_signal_loss_from_batch(
                    self.logits[start:end], chunk, mode=CORRECT_INCORRECT_MARGIN
                )
                * fraction
            )
        torch.testing.assert_close(torch.stack(pieces).sum(), full)

    def test_margin_never_emits_a_slice_that_splits_a_pair(self):
        for start, end, _ in self._slices(
            CORRECT_INCORRECT_MARGIN, target_signal_groups_per_microbatch=1
        ):
            group_ids = self.batch[GROUP_IDS_KEY][start:end]
            self.assertEqual(len(torch.unique(group_ids)), 1)
            self.assertEqual(int((group_ids == group_ids[0]).sum()), 2)

    def test_answer_only_ce_reduces_to_the_same_loss_as_nll_on_its_labels(self):
        expected = causal_token_mean_nll(self.logits, self.labels)
        actual = compute_target_signal_loss_from_batch(
            self.logits, self.batch, mode=ANSWER_ONLY_CE
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
