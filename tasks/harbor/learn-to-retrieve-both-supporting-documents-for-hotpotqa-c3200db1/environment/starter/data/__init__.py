from .base import BaseDataset

def get_dataset(dataset_cfg, model):
    if dataset_cfg.name == "qa":
        # Dispatch: storage=="arrayrecord" uses the indexed path (O(1) resume,
        # pool-uniform permutation). Anything else falls through to streaming.
        if str(getattr(dataset_cfg, "storage", "streaming")) == "arrayrecord":
            from .qa import QADatasetIndexed
            return QADatasetIndexed(tokenizer=model.tokenizer, **dataset_cfg)
        from .qa import QADataset
        return QADataset(model.tokenizer, **dataset_cfg)
    elif dataset_cfg.name == "novelhopqa":
        from .novelhopqa import NovelHopQADataset
        return NovelHopQADataset(model.tokenizer, **dataset_cfg)
    elif dataset_cfg.name == "documents":
        from .documents import DocumentsDataset
        return DocumentsDataset(model.tokenizer, **dataset_cfg)
    elif dataset_cfg.name == "doc_copy":
        # Removable grounding experiment (data/doc_copy.py): target = the positive document.
        from .doc_copy import DocCopyDataset
        return DocCopyDataset(model.tokenizer, **dataset_cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset_cfg.name}")