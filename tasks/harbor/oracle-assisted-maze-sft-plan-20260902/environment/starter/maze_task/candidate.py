"""Starter method: ordinary canonical-path supervised fine-tuning."""


def build_examples(train_groups):
    """Return (catalog_id, weight, is_negative) selections."""
    return [(example["id"], 1.0, False) for group in train_groups
            for example in group["catalog"] if example["kind"] == "canonical"]
