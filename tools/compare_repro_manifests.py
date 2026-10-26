#!/usr/bin/env python3
"""Compare manifests from two DWGC training runs.

Usage: python tools/compare_repro_manifests.py path/to/runA/repro_manifest.json path/to/runB/repro_manifest.json
"""
import json
import sys
from pathlib import Path


def main(a, b):
    left = json.loads(Path(a).read_text(encoding='utf-8'))
    right = json.loads(Path(b).read_text(encoding='utf-8'))
    essential = ['seed', 'deterministic', 'torch', 'numpy', 'cuda_runtime',
                 'cudnn', 'gpus', 'transformers', 'peft', 'scikit-learn',
                 'dataset_sha256', 'word_embedding_sha256',
                 'CUBLAS_WORKSPACE_CONFIG', 'source_sha256']
    print('=== Reproducibility environment differences ===')
    mismatch = False
    for k in essential:
        if left.get(k) != right.get(k):
            mismatch = True
            print(f'{k}:\n  A={left.get(k)}\n  B={right.get(k)}')
    excluded = {'model_id', 'checkpoints', 'des'}
    la = left.get('args', {})
    ra = right.get('args', {})
    for k in sorted(set(la) | set(ra)):
        if k not in excluded and la.get(k) != ra.get(k):
            mismatch = True
            print(f'args.{k}: A={la.get(k)!r}, B={ra.get(k)!r}')
    if not mismatch:
        print('No manifest differences found (pretrained GPT-2 cache weights are not hashed).')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: python tools/compare_repro_manifests.py A/repro_manifest.json B/repro_manifest.json')
    main(sys.argv[1], sys.argv[2])
