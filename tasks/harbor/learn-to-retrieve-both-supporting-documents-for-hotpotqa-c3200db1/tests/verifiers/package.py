"""Build only approved public assets; never copy the reference task wholesale."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import requests


REVISION = '30b0a37ccaaa32f332884b96992754e246e48c5f'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    evaluator = build / 'evaluator'
    previous = build / 'revision-context/current-task/tests'
    public = evaluator / 'public_tests'
    report = {}
    for name, source, destination, files in [
        ('train', previous / 'public_tests/assets/hotpotqa/train', public / 'assets/hotpotqa/train',
         ['corpus.jsonl', 'questions.jsonl']),
        ('public', previous / 'public_tests/public_panel', public / 'public_panel',
         ['corpus.jsonl', 'queries.jsonl', 'gold.jsonl', 'slices.jsonl']),
        ('private', previous / 'private_panel', evaluator / 'private_panel',
         ['corpus.jsonl', 'queries.jsonl', 'gold.jsonl', 'slices.jsonl'])]:
        destination.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((source / 'manifest.json').read_text())
        manifest = {key: value for key, value in manifest.items()
                    if key not in ('embedding_identity', 'embedding', 'negatives', 'schemas')}
        manifest['hashes'] = {filename: manifest['hashes'][filename] for filename in files}
        for filename in files:
            shutil.copyfile(source / filename, destination / filename)
            assert digest(destination / filename) == manifest['hashes'][filename]
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        report[name] = dict(questions=manifest['questions'], paragraphs=manifest['paragraphs'],
                            hashes=manifest['hashes'], unchanged_membership=True)
    model = public / 'assets/models/bert-tiny'
    model.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for filename in ('config.json', 'vocab.txt', 'model.safetensors', 'README.md'):
        target = model / filename
        if not target.exists():
            url = f'https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/{REVISION}/{filename}'
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            target.write_bytes(response.content)
        hashes[filename] = digest(target)
    (model / 'asset_manifest.json').write_text(json.dumps(dict(repo='google/bert_uncased_L-2_H-128_A-2',
                                                              revision=REVISION, license='Apache-2.0', hashes=hashes), indent=2))
    for folder, filenames in {'utils': ['model.py', 'oracle.py', 'process.py', 'public_checkpoint.py'],
                              'verifiers': ['runner.py', 'inference.py', 'public_run.py']}.items():
        (public / folder).mkdir(exist_ok=True)
        for filename in filenames:
            shutil.copyfile(evaluator / folder / filename, public / folder / filename)
    (public / 'incumbent').mkdir(exist_ok=True)
    for filename in ('training.py', 'train_retriever.sh'):
        shutil.copyfile(build / 'starter-overlay' / filename, public / 'incumbent' / filename)
    report['model'] = hashes
    old = json.loads((previous / 'private_panel/construction.json').read_text())
    report['overlap_paragraphs'] = old['overlaps']
    report['exclusions'] = {key: old[key] for key in ('exclusions', 'eligible_training', 'training_groups_excluded', 'partition')}
    (evaluator / 'asset_report.json').write_text(json.dumps(report, indent=2))
    print('Packaged exact text splits, pretrained Tiny, and public dependency closure.')


if __name__ == '__main__':
    main()
