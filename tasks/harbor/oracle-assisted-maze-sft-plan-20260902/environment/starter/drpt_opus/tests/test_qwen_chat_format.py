import json
import unittest

from SFT.data.chat_format import (
    canonicalize_messages_and_tools,
    encode_assistant_only,
)


class _QwenTokenizerStub:
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
            matched = False
            for marker, token_id in self._special.items():
                if text.startswith(marker, index):
                    result.append(token_id)
                    index += len(marker)
                    matched = True
                    break
            if not matched:
                result.append(ord(text[index]) + 100)
                index += 1
        return result

    def decode(self, ids):
        chars = []
        for token_id in ids:
            if token_id in self._special.values():
                continue
            chars.append(chr(token_id - 100))
        return "".join(chars)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        del kwargs
        rendered = ""
        if tools:
            rendered += "<|im_start|>system\nTOOLS:" + json.dumps(tools) + "<|im_end|>\n"
        for message in messages:
            role = message["role"]
            if role == "tool":
                role = "user"
                content = "<tool_response>" + message["content"] + "</tool_response>"
            else:
                content = message.get("content", "")
            rendered += f"<|im_start|>{role}\n{content}"
            for call in message.get("tool_calls", []):
                rendered += "<tool_call>" + json.dumps(call["function"], sort_keys=True)
                rendered += "</tool_call>"
            rendered += "<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return self.encode(rendered) if tokenize else rendered


class QwenChatFormatTests(unittest.TestCase):
    def test_dolci_tool_fields_are_canonicalized(self):
        messages, tools = canonicalize_messages_and_tools({
            "messages": [
                {
                    "role": "system",
                    "content": "use tools",
                    "functions": [{"name": "lookup", "parameters": {"type": "object"}}],
                },
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": None,
                    "function_calls": '{"name":"lookup","arguments":{"q":"x"}}',
                },
                {"role": "environment", "content": {"value": 3}},
                {"role": "assistant", "content": "done"},
            ]
        })

        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "lookup")
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["content"], '{"value": 3}')

    def test_every_assistant_turn_and_tool_call_is_supervised(self):
        tokenizer = _QwenTokenizerStub()
        encoded = encode_assistant_only({
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": None,
                    "function_calls": [{"name": "lookup", "arguments": {"q": "x"}}],
                },
                {"role": "environment", "content": "private tool result"},
                {"role": "assistant", "content": "final answer"},
            ]
        }, tokenizer, max_seq_length=4096)

        labels = encoded["labels"].tolist()
        labelled = tokenizer.decode([token for token in labels if token != -100])
        self.assertIn("lookup", labelled)
        self.assertIn("final answer", labelled)
        self.assertNotIn("question", labelled)
        self.assertNotIn("private tool result", labelled)
        self.assertTrue(encoded["_tokenization_assistant_end_retained"])

    def test_invalid_function_call_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            canonicalize_messages_and_tools({
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": None, "function_calls": "not-json"},
                ]
            })


class CachedQwenTokenizerGoldenTests(unittest.TestCase):
    """Golden spans for the pinned tokenizer, skipped when it is not cached."""

    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer

            from SFT.data.dolci32k.profile import MODEL_PROFILES

            model_profile = MODEL_PROFILES["qwen3_4b"]

            cls.tokenizer = AutoTokenizer.from_pretrained(
                model_profile["tokenizer_name"],
                revision=model_profile["tokenizer_revision"],
                local_files_only=True,
            )
        except (ImportError, OSError, ValueError) as exc:
            raise unittest.SkipTest(
                f"pinned Qwen tokenizer is not available in the local cache: {exc}"
            ) from exc

    def _encode(self, example, expected_total, expected_supervised):
        encoded = encode_assistant_only(
            example, self.tokenizer, max_seq_length=4096
        )
        labels = encoded["labels"]
        supervised = labels[labels != -100]
        self.assertEqual(len(encoded["input_ids"]), expected_total)
        self.assertEqual(len(supervised), expected_supervised)
        self.assertTrue(encoded["_tokenization_assistant_end_retained"])
        return self.tokenizer.decode(
            supervised.tolist(), skip_special_tokens=False
        ), labels

    def test_reasoning_turn_supervises_think_and_answer_only(self):
        supervised, labels = self._encode(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Solve 2 + 2. Show brief reasoning.",
                    },
                    {
                        "role": "assistant",
                        "content": "<think>Two plus two equals four.</think>\n\n4",
                    },
                ]
            },
            expected_total=33,
            expected_supervised=12,
        )
        self.assertIn("Two plus two equals four", supervised)
        self.assertIn("\n\n4", supervised)
        self.assertNotIn("Solve 2 + 2", supervised)
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assertEqual(int((labels == end_id).sum()), 1)

    def test_multi_turn_supervises_every_assistant_terminator(self):
        supervised, labels = self._encode(
            {
                "messages": [
                    {"role": "user", "content": "Name a primary color."},
                    {"role": "assistant", "content": "Red."},
                    {"role": "user", "content": "Name another one."},
                    {"role": "assistant", "content": "Blue."},
                ]
            },
            expected_total=37,
            expected_supervised=10,
        )
        self.assertIn("Red.", supervised)
        self.assertIn("Blue.", supervised)
        self.assertNotIn("Name another one", supervised)
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assertEqual(int((labels == end_id).sum()), 2)

    def test_tool_call_is_supervised_but_tool_result_is_masked(self):
        supervised, labels = self._encode(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Use tools when helpful.",
                        "functions": [
                            {
                                "name": "lookup",
                                "description": "Look up a value",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "key": {"type": "string"}
                                    },
                                    "required": ["key"],
                                },
                            }
                        ],
                    },
                    {"role": "user", "content": "Look up alpha."},
                    {
                        "role": "assistant",
                        "content": None,
                        "function_calls": [
                            {
                                "name": "lookup",
                                "arguments": {"key": "alpha"},
                            }
                        ],
                    },
                    {
                        "role": "environment",
                        "content": "DO_NOT_LABEL_TOOL_OUTPUT",
                    },
                    {"role": "assistant", "content": "Lookup completed."},
                ]
            },
            expected_total=199,
            expected_supervised=27,
        )
        self.assertIn("<tool_call>", supervised)
        self.assertIn('"name": "lookup"', supervised)
        self.assertIn("Lookup completed.", supervised)
        self.assertNotIn("DO_NOT_LABEL_TOOL_OUTPUT", supervised)
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assertEqual(int((labels == end_id).sum()), 2)


if __name__ == "__main__":
    unittest.main()
