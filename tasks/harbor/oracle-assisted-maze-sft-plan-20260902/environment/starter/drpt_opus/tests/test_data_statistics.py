import io
import unittest
from contextlib import redirect_stdout

from datasets import Dataset

from SFT.data.get_train_dataset import encode_data
from SFT.train.data_arguments import get_data_statistics


class _TokenizerStub:
    chat_template = None

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        del tokenize, kwargs
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def encode(self, text, **kwargs):
        del kwargs
        return list(text.encode("utf-8"))


class DataStatisticsTests(unittest.TestCase):
    def test_torch_formatted_text_dataset_does_not_require_torchvision_io(self):
        dataset = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3], [4, 5]],
                "labels": [[-100, 2, 3], [-100, 5]],
            }
        )
        dataset.set_format(type="torch")

        with redirect_stdout(io.StringIO()):
            average = get_data_statistics(dataset, return_avg_length=True)

        self.assertEqual(average, 2.5)
        self.assertEqual(dataset.format["type"], "torch")

    def test_encoded_text_dataset_stays_unformatted_for_trainer_collation(self):
        dataset = Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                    ]
                ]
            }
        )

        encoded = encode_data(
            dataset,
            _TokenizerStub(),
            max_seq_length=64,
            processing_num_workers=1,
            overwrite_cache=True,
        )

        self.assertIsNone(encoded.format["type"])
        self.assertIsInstance(encoded[0]["input_ids"], list)
        self.assertEqual(len(encoded[0]["input_ids"]), len(encoded[0]["labels"]))


if __name__ == "__main__":
    unittest.main()
