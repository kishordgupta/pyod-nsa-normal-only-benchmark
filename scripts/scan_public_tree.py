"""Fail on credential or private-host patterns; never print matched values."""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'github_token': r'(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'cloud_access_key': r'(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])',
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'api_secret': r'(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}',
    'credential_url': r'https?://[^\s/:]+:[^\s/@]+@',
    'user_absolute_path': r'/Users/[^\s/]+/',
    'private_overleaf_link': r'https?://(?:www\.)?overleaf\.com/(?:project|read|[0-9]+)',
    'private_email': r'[A-Za-z0-9._%+-]+@(?:gmail|outlook|hotmail|yahoo)\.com',
}
findings = []
scanned = 0
public_attribution_hits = 0
manifest_path = ROOT / 'provenance/public-copy-manifest.json'
source_files = {item['public_file'] for item in json.loads(manifest_path.read_text())['files']
                if item['source_category'] == 'pinned-pyod-source' and item['bytes_identical']}
for path in sorted(ROOT.rglob('*')):
    parts = path.relative_to(ROOT).parts
    if not path.is_file() or any(p in {'.git', '.venv', 'local_runs', 'reanalysis', '__pycache__', 'cache'} for p in parts):
        continue
    # The scanner's literal regular expressions are documentation, not secrets.
    if path.resolve() == Path(__file__).resolve():
        continue
    try:
        content = path.read_text()
    except UnicodeDecodeError:
        continue
    scanned += 1
    for name, expression in RULES.items():
        if name == 'private_email' and '@' not in content:
            continue
        if re.search(expression, content, flags=re.I if name.endswith('email') else 0):
            relative = str(path.relative_to(ROOT))
            # Attribution emails in byte-identical pinned public upstream source
            # were already publicly distributed; do not rewrite the source.
            if name == 'private_email' and relative in source_files:
                public_attribution_hits += 1
            else:
                findings.append({'file': relative, 'rule': name})
print(json.dumps({'text_files_scanned': scanned, 'findings': findings,
                  'verified_public_upstream_attribution_files': public_attribution_hits}, indent=2))
sys.exit(bool(findings))
