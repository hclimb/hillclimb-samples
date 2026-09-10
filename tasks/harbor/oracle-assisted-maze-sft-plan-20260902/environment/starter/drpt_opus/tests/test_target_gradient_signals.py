from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from SFT.train.target_signal import (
    ANSWER_ONLY_CE,
    CORRECT_INCORRECT_MARGIN,
    NLL,
    REWARD_WEIGHTED_SFT,
    ROLE_CHOSEN,
    ROLE_NEUTRAL,
    ROLE_REJECTED,
    GroupedTargetCollator,
    canonicalize_target_signal_mode,
    causal_token_mean_nll,
    compute_target_signal_loss,
    compute_target_signal_loss_from_batch,
    correct_incorrect_margin_loss,
    grouped_slice_normalization_weight,
    iter_grouped_logical_slices,
    iter_logical_slices,
    model_inputs_from_target_batch,
    response_logp_stats,
    reward_weighted_sft_loss,
    shifted_logits_and_labels,
    shifted_token_counts,
    shifted_token_logps,
    shifted_token_mask,
    slice_normalization_weight,
    slice_target_batch,
    token_mean_normalization_weights,
)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


class TargetSignalModeTests(unittest.TestCase):
    def test_canonical_modes_and_aliases(self) -> None:
        self.assertEqual(canonicalize_target_signal_mode("nll"), NLL)
        self.assertEqual(canonicalize_target_signal_mode("answer-only-ce"), ANSWER_ONLY_CE)
        self.assertEqual(canonicalize_target_signal_mode("margin"), CORRECT_INCORRECT_MARGIN)
        self.assertEqual(canonicalize_target_signal_mode("reward_weighted"), REWARD_WEIGHTED_SFT)
        with self.assertRaisesRegex(ValueError, "unknown target-signal mode"):
            canonicalize_target_signal_mode("not_a_signal")

    def test_answer_only_and_nll_share_exact_loss_math(self) -> None:
        torch.manual_seed(11)
        logits = torch.randn(2, 5, 7, dtype=torch.double)
        answer_labels = torch.tensor(
            [
                [-100, -100, -100, 3, 4],
                [-100, -100, 2, -100, 1],
            ]
        )
        nll = compute_target_signal_loss(logits, answer_labels, mode=NLL)
        answer_only = compute_target_signal_loss(logits, answer_labels, mode=ANSWER_ONLY_CE)
        torch.testing.assert_close(answer_only, nll, rtol=0, atol=0)


class GroupedTargetCollatorTests(unittest.TestCase):
    def test_flattens_candidates_but_preserves_prompt_groups(self) -> None:
        collator = GroupedTargetCollator(pad_token_id=0, pad_to_multiple_of=4)
        features = [
            {
                "candidates": [
                    {
                        "input_ids": [1, 2, 3],
                        "labels": [-100, 2, 3],
                        "role": "correct",
                        "reward": 1,
                    },
                    {
                        "input_ids": [1, 6],
                        "labels": [-100, 6],
                        "role": "incorrect",
                        "reward": 0,
                    },
                ]
            },
            {
                "input_ids": [7, 8, 9, 10],
                "labels": [-100, -100, 9, 10],
                "role": "neutral",
                "reward": 0.25,
            },
        ]

        batch = collator(features)

        self.assertEqual(tuple(batch["input_ids"].shape), (3, 4))
        self.assertEqual(tuple(batch["labels"].shape), (3, 4))
        self.assertEqual(tuple(batch["attention_mask"].shape), (3, 4))
        self.assertEqual(batch["group_ids"].tolist(), [0, 0, 1])
        self.assertEqual(
            batch["roles"].tolist(), [ROLE_CHOSEN, ROLE_REJECTED, ROLE_NEUTRAL]
        )
        torch.testing.assert_close(batch["rewards"], torch.tensor([1.0, 0.0, 0.25]))
        self.assertEqual(batch["input_ids"][1].tolist(), [1, 6, 0, 0])
        self.assertEqual(batch["labels"][1].tolist(), [-100, 6, -100, -100])
        self.assertEqual(batch["attention_mask"][1].tolist(), [1, 1, 0, 0])

    def test_custom_answer_label_field_and_left_padding(self) -> None:
        collator = GroupedTargetCollator(
            pad_token_id=9,
            label_key="answer_labels",
            padding_side="left",
        )
        batch = collator(
            [
                {"input_ids": [1, 2], "answer_labels": [-100, 2]},
                {"input_ids": [3, 4, 5], "answer_labels": [-100, -100, 5]},
            ]
        )
        self.assertEqual(batch["input_ids"][0].tolist(), [9, 1, 2])
        self.assertEqual(batch["labels"][0].tolist(), [-100, -100, 2])
        self.assertEqual(batch["attention_mask"][0].tolist(), [0, 1, 1])

    def test_requires_pre_masked_labels(self) -> None:
        collator = GroupedTargetCollator(pad_token_id=0)
        with self.assertRaisesRegex(KeyError, "labels"):
            collator([{"input_ids": [1, 2]}])

    def test_collated_batch_runs_through_toy_model_and_backpropagates(self) -> None:
        torch.manual_seed(3)
        collator = GroupedTargetCollator(pad_token_id=0)
        batch = collator(
            [
                {
                    "candidates": [
                        {
                            "input_ids": [1, 2, 3],
                            "labels": [-100, 2, 3],
                            "reward": 1.0,
                        },
                        {
                            "input_ids": [1, 4],
                            "labels": [-100, 4],
                            "reward": 0.5,
                        },
                    ]
                }
            ]
        )
        model = TinyCausalLM()
        outputs = model(**model_inputs_from_target_batch(batch))
        loss = compute_target_signal_loss_from_batch(
            outputs.logits, batch, mode=REWARD_WEIGHTED_SFT
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.embedding.weight.grad)
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0.0)


class ShiftedTokenHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.logits = torch.randn(2, 5, 6, dtype=torch.double)
        # The first label is deliberately out of vocabulary: causal shifting
        # must discard it before validation/gathering.
        self.labels = torch.tensor(
            [
                [999, 1, 2, -100, 4],
                [999, -100, 3, 2, -100],
            ]
        )

    def test_shift_shapes_mask_counts_and_selected_logps(self) -> None:
        shifted_logits, shifted_labels = shifted_logits_and_labels(self.logits, self.labels)
        self.assertEqual(tuple(shifted_logits.shape), (2, 4, 6))
        self.assertEqual(tuple(shifted_labels.shape), (2, 4))
        expected_mask = shifted_labels != -100
        torch.testing.assert_close(shifted_token_mask(self.labels), expected_mask)
        self.assertEqual(shifted_token_counts(self.labels).tolist(), [3, 2])

        safe_labels = shifted_labels.masked_fill(~expected_mask, 0)
        expected_logps = (
            F.log_softmax(shifted_logits, dim=-1)
            .gather(-1, safe_labels.unsqueeze(-1))
            .squeeze(-1)
            .masked_fill(~expected_mask, 0)
        )
        torch.testing.assert_close(
            shifted_token_logps(self.logits, self.labels), expected_logps
        )

    def test_nll_is_exact_global_shifted_cross_entropy(self) -> None:
        shifted_logits = self.logits[:, :-1]
        shifted_labels = self.labels[:, 1:]
        expected = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        actual = causal_token_mean_nll(self.logits, self.labels)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class AlternativeTargetLossTests(unittest.TestCase):
    def test_margin_uses_per_response_mean_logp_and_requested_formula(self) -> None:
        torch.manual_seed(23)
        logits = torch.randn(4, 5, 7, dtype=torch.double, requires_grad=True)
        labels = torch.tensor(
            [
                [-100, 1, 2, 3, -100],
                [-100, 1, 2, -100, -100],
                [-100, 4, 5, 6, 1],
                [-100, 4, -100, 6, 1],
            ]
        )
        group_ids = torch.tensor([0, 0, 1, 1])
        roles = torch.tensor([ROLE_CHOSEN, ROLE_REJECTED, ROLE_REJECTED, ROLE_CHOSEN])
        beta = 1.7
        margin = 0.35

        stats = response_logp_stats(logits, labels)
        deltas = torch.stack(
            [
                stats.mean_logps[0] - stats.mean_logps[1],
                stats.mean_logps[3] - stats.mean_logps[2],
            ]
        )
        expected = F.softplus(beta * (margin - deltas)).mean()
        actual = correct_incorrect_margin_loss(
            logits,
            labels,
            group_ids,
            roles,
            beta=beta,
            margin=margin,
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_margin_rejects_ambiguous_pairs(self) -> None:
        logits = torch.zeros(2, 3, 4)
        labels = torch.tensor([[-100, 1, 2], [-100, 2, 1]])
        with self.assertRaisesRegex(ValueError, "exactly one chosen and one rejected"):
            correct_incorrect_margin_loss(
                logits,
                labels,
                group_ids=[0, 0],
                roles=[ROLE_CHOSEN, ROLE_CHOSEN],
            )

    def test_reward_loss_is_weighted_token_mean(self) -> None:
        torch.manual_seed(29)
        logits = torch.randn(3, 5, 8, dtype=torch.double)
        labels = torch.tensor(
            [
                [-100, 1, 2, 3, 4],
                [-100, 2, -100, 5, -100],
                [-100, 6, 7, 1, -100],
            ]
        )
        rewards = torch.tensor([2.0, 0.5, 0.0], dtype=torch.double)

        shifted_logits = logits[:, :-1]
        shifted_labels = labels[:, 1:]
        mask = shifted_labels != -100
        safe_labels = shifted_labels.masked_fill(~mask, 0)
        token_nll = -(
            F.log_softmax(shifted_logits, dim=-1)
            .gather(-1, safe_labels.unsqueeze(-1))
            .squeeze(-1)
        )
        expected = (
            (token_nll * mask * rewards[:, None]).sum()
            / (mask * rewards[:, None]).sum()
        )
        actual = reward_weighted_sft_loss(logits, labels, rewards)
        torch.testing.assert_close(actual, expected)

    def test_all_zero_reward_mass_is_rejected(self) -> None:
        logits = torch.zeros(2, 3, 4)
        labels = torch.tensor([[-100, 1, 2], [-100, 2, 1]])
        with self.assertRaisesRegex(ValueError, "positive reward mass"):
            reward_weighted_sft_loss(logits, labels, [0.0, 0.0])


class LogicalMicrobatchTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.logits = torch.randn(5, 6, 9, dtype=torch.double)
        self.labels = torch.tensor(
            [
                [-100, 1, 2, 3, 4, 5],
                [-100, -100, -100, 2, 3, 4],
                [-100, 3, -100, -100, 4, -100],
                [-100, 4, 5, 6, -100, -100],
                [-100, 7, 8, 1, 2, -100],
            ]
        )

    def test_nll_recomposes_exactly_from_row_microbatches(self) -> None:
        full_loss = causal_token_mean_nll(self.logits, self.labels)
        norm_weights = token_mean_normalization_weights(self.labels)
        pieces = []
        self.assertEqual(list(iter_logical_slices(5, 2)), [(0, 2), (2, 4), (4, 5)])
        for start, end in iter_logical_slices(5, 2):
            local_loss = causal_token_mean_nll(
                self.logits[start:end], self.labels[start:end]
            )
            pieces.append(local_loss * slice_normalization_weight(norm_weights, start, end))
        torch.testing.assert_close(torch.stack(pieces).sum(), full_loss)

    def test_reward_loss_recomposes_exactly_from_row_microbatches(self) -> None:
        rewards = torch.tensor([1.0, 0.25, 2.0, 0.5, 1.5], dtype=torch.double)
        full_loss = reward_weighted_sft_loss(self.logits, self.labels, rewards)
        norm_weights = token_mean_normalization_weights(self.labels, rewards)
        pieces = []
        for start, end in iter_logical_slices(5, 2):
            local_loss = reward_weighted_sft_loss(
                self.logits[start:end],
                self.labels[start:end],
                rewards[start:end],
            )
            pieces.append(local_loss * slice_normalization_weight(norm_weights, start, end))
        torch.testing.assert_close(torch.stack(pieces).sum(), full_loss)

    def test_group_slices_never_split_pairs_and_recompose_margin(self) -> None:
        # Candidate counts per prompt are 2, 3, and 2.
        group_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        slices = list(iter_grouped_logical_slices(group_ids, groups_per_microbatch=2))
        self.assertEqual(slices, [(0, 5), (5, 7)])
        self.assertAlmostEqual(grouped_slice_normalization_weight(group_ids, 0, 5), 2 / 3)
        self.assertAlmostEqual(grouped_slice_normalization_weight(group_ids, 5, 7), 1 / 3)

        torch.manual_seed(37)
        logits = torch.randn(7, 4, 6, dtype=torch.double)
        labels = torch.tensor([[-100, 1, 2, 3]] * 7)
        roles = torch.tensor(
            [
                ROLE_CHOSEN,
                ROLE_REJECTED,
                ROLE_CHOSEN,
                ROLE_REJECTED,
                ROLE_NEUTRAL,
                ROLE_CHOSEN,
                ROLE_REJECTED,
            ]
        )
        full_loss = correct_incorrect_margin_loss(logits, labels, group_ids, roles)
        pieces = []
        batch = {
            "input_ids": torch.ones(7, 4, dtype=torch.long),
            "labels": labels,
            "group_ids": group_ids,
            "roles": roles,
            "rewards": torch.ones(7),
            "run_name": "kept",
        }
        for start, end in slices:
            local_batch = slice_target_batch(batch, start, end)
            local_loss = correct_incorrect_margin_loss(
                logits[start:end],
                local_batch["labels"],
                local_batch["group_ids"],
                local_batch["roles"],
            )
            self.assertEqual(local_batch["run_name"], "kept")
            weight = grouped_slice_normalization_weight(group_ids, start, end)
            pieces.append(local_loss * weight)
        torch.testing.assert_close(torch.stack(pieces).sum(), full_loss)

    def test_noncontiguous_group_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous run"):
            list(iter_grouped_logical_slices([0, 1, 0], groups_per_microbatch=1))


if __name__ == "__main__":
    unittest.main()
