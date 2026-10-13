"""Check published evidence hashes and the full-grid numerical invariants."""
from collections import Counter
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "provenance/public-copy-manifest.json").read_text())
for item in manifest["files"]:
    path = ROOT / item["public_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["public_sha256"]:
        raise ValueError(f"Published hash mismatch: {item['public_file']}")
files = sorted((ROOT / "benchmark/odds_oneclass/full_retention/raw").glob("*.jsonl"))
assert len(files) == 680
rows = [json.loads(line) for path in files for line in path.read_text().splitlines()]
assert len(rows) == 38080
keys = {(r['dataset'], r['seed'], r['representation'], r['retention_fraction'], r['method']) for r in rows}
assert len(keys) == len(rows)
assert len({r['method'] for r in rows}) == 56
assert len({r['dataset'] for r in rows}) == 17
assert all(r['fit_outliers'] == 0 for r in rows)
assert all(len(path.read_text().splitlines()) == 56 for path in files)
assert Counter(r['status'] for r in rows) == {
    'success': 32510, 'timeout': 1548, 'requires_labeled_outliers': 1360,
    'error': 1301, 'requires_1d': 680, 'requires_2d_or_3d': 340,
    'requires_2d': 340, 'build_error': 1}
print(json.dumps({'published_hashes_verified': len(manifest['files']),
                  'full_retention_cells': len(files), 'raw_records': len(rows),
                  'all_training_outlier_counts_zero': True}, indent=2))
