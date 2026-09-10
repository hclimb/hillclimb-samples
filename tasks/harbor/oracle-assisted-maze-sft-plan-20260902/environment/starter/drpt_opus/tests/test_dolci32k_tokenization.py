"""Network-free contracts for Dolci32K tokenizer diagnostics and caches."""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from SFT.data.chat_format import (
    SUPERVISION_TRUNCATION_POLICY,
    SUPERVISION_TRUNCATION_POLICY_VERSION,
    encode_assistant_only,
)
from SFT.data.dolci32k.tokenization import (
    TokenizationTruncationWarning,
    ZeroSupervisionError,
    assert_no_zero_supervision,
    diagnose_record,
    load_tokenization_report,
    persist_tokenization_report,
    profile_jsonl_artifacts,
    profile_tokenizer_profiles,
    zero_supervision_report,
)


class _QwenTokenizerStub:
    """One-character tokenizer with Qwen delimiters and deterministic lengths."""

    chat_template = "<|im_start|>...<|im_end|>"
    unk_token_id = -1
    _special = {"<|im_start|>": 1, "<|im_end|>": 2}

    def convert_tokens_to_ids(self, token):
        return self._special.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        result = []
        index = 0
        while index < len(text):
            for marker, token_id in self._special.items():
                if text.startswith(marker, index):
                    result.append(token_id)
                    index += len(marker)
                    break
            else:
                result.append(ord(text[index]) + 100)
                index += 1
        return result

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        del tools, kwargs
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message.get('content', '')}<|im_end|>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return self.encode(rendered) if tokenize else rendered


def _record(record_id: str, answer: str, *, prompt: str = "u") -> dict:
    return {
        "id": record_id,
        "source_dataset": "fixture-source",
        "domain": "Chat",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class Dolci32KTokenizationTests(unittest.TestCase):
    def test_linear_quantiles_and_max_are_saved_per_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "pool.jsonl"
            _write_jsonl(
                artifact,
                [
                    _record("row-1", "a"),
                    _record("row-2", "aaa"),
                    _record("row-3", "aaaaa"),
                    _record("row-4", "aaaaaaa"),
                ],
            )
            result = profile_jsonl_artifacts(
                {"instruction_train": artifact},
                raw_build_id="raw-quantiles",
                tokenizer_profile={"name": "stub"},
                tokenizer=_QwenTokenizerStub(),
                cache_root=root / "cache",
                max_seq_len=100,
                emit_warnings=False,
            )

            lengths = result["statistics"]["artifacts"]["instruction_train"][
                "token_length"
            ]
            self.assertEqual(lengths["p50"], 24.0)
            self.assertAlmostEqual(lengths["p90"], 26.4)
            self.assertAlmostEqual(lengths["p99"], 26.94)
            self.assertEqual(lengths["max"], 27)
            self.assertTrue(Path(result["parquet_path"]).is_file())
            self.assertEqual(
                result["truncation_policy"], SUPERVISION_TRUNCATION_POLICY
            )
            self.assertEqual(
                result["truncation_policy_version"],
                SUPERVISION_TRUNCATION_POLICY_VERSION,
            )

    def test_more_than_five_percent_over_length_emits_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "pool.jsonl"
            rows = [_record(f"short-{index}", "a") for index in range(9)]
            rows.append(_record("long", "a" * 20))
            _write_jsonl(artifact, rows)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = profile_jsonl_artifacts(
                    {"instruction_train": artifact},
                    raw_build_id="raw-warning",
                    tokenizer_profile={"name": "stub"},
                    tokenizer=_QwenTokenizerStub(),
                    cache_root=root / "cache",
                    max_seq_len=30,
                )

            stats = result["statistics"]["artifacts"]["instruction_train"]
            self.assertEqual(stats["exceeds_max_seq_len"]["count"], 1)
            self.assertEqual(stats["exceeds_max_seq_len"]["fraction"], 0.1)
            self.assertEqual(
                stats["over_length_by_source_dataset"]["fixture-source"]["count"],
                1,
            )
            self.assertEqual(len(result["warnings"]), 1)
            self.assertTrue(
                any(
                    item.category is TokenizationTruncationWarning
                    and "WARNING:" in str(item.message)
                    for item in caught
                )
            )

    def test_zero_label_right_truncation_uses_shared_deterministic_fallback(self):
        tokenizer = _QwenTokenizerStub()
        partial = diagnose_record(
            _record("partial", "a" * 50), tokenizer, max_seq_len=25
        )
        zero = diagnose_record(
            _record("zero", "a", prompt="u" * 100),
            tokenizer,
            max_seq_len=20,
        )

        self.assertTrue(partial["assistant_truncated"])
        self.assertGreater(partial["assistant_tokens_retained"], 0)
        self.assertFalse(partial["zero_supervised_after_right_truncation"])
        self.assertFalse(partial["supervision_preserving_fallback_applied"])
        self.assertTrue(zero["assistant_truncated_by_right_truncation"])
        self.assertEqual(zero["assistant_tokens_retained_by_right_truncation"], 0)
        self.assertTrue(zero["zero_supervised_after_right_truncation"])
        self.assertTrue(zero["supervision_preserving_fallback_applied"])
        self.assertGreater(zero["truncation_window_start"], 0)
        self.assertGreater(zero["assistant_tokens_retained"], 0)
        self.assertFalse(zero["assistant_truncated"])
        self.assertFalse(zero["zero_supervised_after_truncation"])

        encoded = encode_assistant_only(
            _record("zero", "a", prompt="u" * 100), tokenizer, 20
        )
        self.assertEqual(encoded["input_ids"].tolist(), encode_assistant_only(
            _record("zero", "a", prompt="u" * 100), tokenizer, 20
        )["input_ids"].tolist())
        self.assertEqual(
            int((encoded["labels"] != -100).sum()),
            zero["assistant_tokens_retained"],
        )
        self.assertTrue(
            encoded["_tokenization_supervision_preserving_fallback_applied"]
        )
        self.assertFalse(encoded["_tokenization_zero_supervised_after_truncation"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "pool.jsonl"
            _write_jsonl(artifact, [_record("zero", "a", prompt="u" * 100)])
            result = profile_jsonl_artifacts(
                {"instruction_train": artifact},
                raw_build_id="raw-zero",
                tokenizer_profile={"name": "stub"},
                tokenizer=tokenizer,
                cache_root=root / "cache",
                max_seq_len=20,
                emit_warnings=False,
            )
            stats = result["statistics"]["artifacts"]["instruction_train"]
            self.assertEqual(
                stats["zero_supervised_after_right_truncation"]["ids"], ["zero"]
            )
            self.assertEqual(
                stats["supervision_preserving_fallback"]["ids"], ["zero"]
            )
            self.assertEqual(
                stats["zero_supervised_after_truncation"]["count"], 0
            )
            self.assertEqual(zero_supervision_report(result)["ids"], [])
            assert_no_zero_supervision(result)

    def test_final_zero_supervision_remains_fail_closed(self):
        final_zero_row = {
            "artifact": "instruction_train",
            "id": "still-zero",
            "source_dataset": "fixture-source",
            "zero_supervised_after_truncation": True,
        }
        with mock.patch(
            "SFT.data.dolci32k.tokenization.read_tokenization_cache",
            return_value=[final_zero_row],
        ):
            report = zero_supervision_report("unused.parquet")
            self.assertEqual(report["ids"], ["still-zero"])
            with self.assertRaises(ZeroSupervisionError):
                assert_no_zero_supervision("unused.parquet")

    def test_identical_qwen_aliases_reuse_one_physical_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "pool.jsonl"
            _write_jsonl(artifact, [_record("row", "answer")])
            result = profile_tokenizer_profiles(
                {"instruction_train": artifact},
                raw_build_id="raw-aliases",
                tokenizer_profiles={
                    "qwen3_4b": {"tokenizer_repo": "fixture/4b"},
                    "qwen3_8b": {"tokenizer_repo": "fixture/8b"},
                },
                tokenizer_factory=lambda _: _QwenTokenizerStub(),
                cache_root=root / "cache",
                max_seq_len=100,
                emit_warnings=False,
            )

            self.assertEqual(len(result["physical_caches"]), 1)
            four = result["profiles"]["qwen3_4b"]
            eight = result["profiles"]["qwen3_8b"]
            self.assertEqual(four["tokenizer_fingerprint"], eight["tokenizer_fingerprint"])
            self.assertEqual(four["parquet_path"], eight["parquet_path"])
            self.assertFalse(four["cache_reused"])
            self.assertTrue(eight["cache_reused"])

    def test_persisted_alias_report_validates_physical_cache_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "pool.jsonl"
            _write_jsonl(artifact, [_record("row", "answer")])
            report = profile_tokenizer_profiles(
                {"general/instruction_32k/train": artifact},
                raw_build_id="a" * 64,
                tokenizer_profiles={"qwen3_4b": {"tokenizer_repo": "fixture/4b"}},
                tokenizer_factory=lambda _: _QwenTokenizerStub(),
                cache_root=root / "cache",
                max_seq_len=100,
                emit_warnings=False,
            )
            report_path = persist_tokenization_report(report, data_dir=root)
            loaded = load_tokenization_report(
                report_path, raw_build_id="a" * 64, max_seq_len=100
            )
            self.assertEqual(set(loaded["profiles"]), {"qwen3_4b"})

            parquet = Path(loaded["profiles"]["qwen3_4b"]["parquet_path"])
            with parquet.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                load_tokenization_report(report_path)


if __name__ == "__main__":
    unittest.main()
