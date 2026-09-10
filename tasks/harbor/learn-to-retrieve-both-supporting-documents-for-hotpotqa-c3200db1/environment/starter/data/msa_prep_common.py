DATASET_SPECS = {
    "2wikimultihopqa": {
        "doc_repo": "vm2825/msa-2wikimultihopqa-docs",
        "qa_repo": "vm2825/msa-2wikimultihopqa-qa",
        "doc_separator": "\n||||\n",
    },
    "dureader": {
        "doc_repo": "vm2825/msa-dureader-docs",
        "qa_repo": "vm2825/msa-dureader-qa",
        "doc_separator": "\n||||\n",
    },
    "hotpotqa": {
        "doc_repo": "vm2825/msa-hotpotqa-docs",
        "qa_repo": "vm2825/msa-hotpotqa-qa",
        "doc_separator": "\n||||\n",
    },
    "msmarco_v1": {
        "doc_repo": "vm2825/msa-msmarco-v1-docs",
        "qa_repo": "vm2825/msa-msmarco-v1-qa",
        "doc_separator": "\n||||\n",
    },
    "musique": {
        "doc_repo": "vm2825/msa-musique-docs",
        "qa_repo": "vm2825/msa-musique-qa",
        "doc_separator": "\n||||\n",
    },
    "narrativeqa": {
        "doc_repo": "vm2825/msa-narrativeqa-docs",
        "qa_repo": "vm2825/msa-narrativeqa-qa",
        "doc_separator": "\n||||\n",
    },
    "natural_questions": {
        "doc_repo": "vm2825/msa-natural-questions-docs",
        "qa_repo": "vm2825/msa-natural-questions-qa",
        "doc_separator": "\n||||\n",
    },
    "popqa": {
        "doc_repo": "vm2825/msa-popqa-docs",
        "qa_repo": "vm2825/msa-popqa-qa",
        "doc_separator": "\n||||\n",
    },
    "triviaqa_10m": {
        "doc_repo": "vm2825/msa-triviaqa-10m-docs",
        "qa_repo": "vm2825/msa-triviaqa-10m-qa",
        "doc_separator": "\n||||\n",
    },
}


def split_pos_doc(text: str, separator: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(separator) if part.strip()]
