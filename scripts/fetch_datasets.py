"""Download the 17 immutable benchmark files and check their saved SHA-256."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "benchmark/odds_oneclass/dataset_manifest.json").read_text())
destination = ROOT / "vendor/pyod/notebooks/data"
destination.mkdir(parents=True, exist_ok=True)
for item in manifest["datasets"]:
    path = destination / item["filename"]
    if not path.exists():
        url = ("https://raw.githubusercontent.com/yzhao062/pyod/"
               + manifest["source"]["original_data_commit"]
               + "/notebooks/data/" + item["filename"])
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Downloaded SHA-256 mismatch: {item['name']}")
        temporary = path.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"Local SHA-256 mismatch: {item['name']}")
    print(f"Verified {item['name']}")
