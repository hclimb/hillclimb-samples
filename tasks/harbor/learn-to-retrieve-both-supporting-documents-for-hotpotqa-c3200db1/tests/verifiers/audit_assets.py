import hashlib
import json
from pathlib import Path


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def audit(root):
    texts, ids = {}, {}
    for name, directory, expected_docs, expected_queries in [
        ('train', root / 'public_tests/assets/hotpotqa/train', 200000, 16384),
        ('public', root / 'public_tests/public_panel', 60000, 3072),
        ('private', root / 'private_panel', 60000, 3072)]:
        manifest = json.loads((directory / 'manifest.json').read_text())
        for filename, digest in manifest['hashes'].items():
            with (directory / filename).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
        corpus = read_rows(directory / 'corpus.jsonl')
        queries = read_rows(directory / ('questions.jsonl' if name == 'train' else 'queries.jsonl'))
        assert len(corpus) == expected_docs and len(queries) == expected_queries
        assert all(row['id'] == index and set(row) == {'id', 'text'} for index, row in enumerate(corpus))
        texts[name] = {row['text'] for row in corpus}
        ids[name] = {row['id'] for row in queries}
        assert len(texts[name]) == expected_docs and len(ids[name]) == expected_queries
        assert [row['id'] for row in queries] == manifest['selected_ids']
        gold = queries if name == 'train' else read_rows(directory / 'gold.jsonl')
        assert {row['id'] for row in gold} == ids[name]
        for row in gold:
            assert len(row['pos_doc_ids']) == len(set(row['pos_doc_ids'])) == 2
            assert all(type(index) is int and 0 <= index < expected_docs for index in row['pos_doc_ids'])
            if name == 'train':
                assert len(row['context_doc_ids']) == len(set(row['context_doc_ids'])) == 10
                assert all(type(index) is int and 0 <= index < expected_docs for index in row['context_doc_ids'])
                assert set(row['pos_doc_ids']) <= set(row['context_doc_ids'])
        if name != 'train':
            assert all(set(row) == {'id', 'question'} for row in queries)
    assert not ids['public'] & ids['private'] and not ids['train'] & (ids['public'] | ids['private'])
    models = root / 'public_tests/assets/models'
    assert {path.name for path in models.iterdir()} == {'bert-tiny'}
    assert {path.name for path in (root / 'public_tests/assets/hotpotqa/train').iterdir()} == {
        'manifest.json', 'corpus.jsonl', 'questions.jsonl'}
    manifest = json.loads((models / 'bert-tiny/asset_manifest.json').read_text())
    for filename, digest in manifest['hashes'].items():
        assert hashlib.sha256((models / 'bert-tiny' / filename).read_bytes()).hexdigest() == digest
    report = dict(valid=1, overlaps={f'{left}/{right}': len(texts[left] & texts[right])
                                   for left, right in [('train', 'public'), ('train', 'private'), ('public', 'private')]})
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    audit(Path(__file__).resolve().parents[1])
