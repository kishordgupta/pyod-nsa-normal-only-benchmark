# Publication changes and evidence preservation

The private originals were not edited. Published files were selected from the
benchmark workspace, not from the author's entire working directory. Virtual
environments, caches, bytecode, unrelated files, temporary process files and
the original ZIP archives were excluded.

Host-specific absolute paths were replaced with repository-relative paths or
generic environment placeholders in logs, traceback messages, reports and
metadata. Published scientific outcomes, dataset/seed/retention/method keys,
labels, class counts, scores, elapsed times and split identity hashes were
preserved. Where redaction changed bytes, metadata digest references were
recomputed for the public copy. `public-copy-manifest.json` records both the
private original SHA-256 and the published SHA-256 without recording private
filenames. These two digests differ only for redacted or relocated files.

The runner, model factory, training split, transforms and score computation
are unchanged. Runnable source defaults in `build_manifest.py` and
`supervised_probe.py` locate `vendor/pyod` relative to the repository.
The dataset-rank script writes to `results` in this repository. The sweep
records the vendored source revision from `SOURCE_PROVENANCE.json` rather
than treating the new benchmark repository's Git revision as PyOD's revision.
Sanitized copies of the pre-relocation scripts are retained under
`provenance/archived-code`.

Restart and watchdog sidecars have digest references recomputed for the public
log/snapshot views, with the original file hashes retained in the publication
manifest. Published historical analysis tables keep their numerical values.
Reanalysis writes separate fresh audits instead of silently replacing the
archived result tables.

The automated publication scan reports filenames and rule names only. It
checks token formats, private keys, credential-bearing URLs, personal email
domains, private Overleaf URLs and personal host paths. A clean scan is
evidence for these checks; it cannot prove the absence of every possible
secret. Repository publication includes only benchmark artifacts and public
upstream software, not account/session state or browser captures.
Public attribution emails in byte-identical pinned upstream source files are
reported as verified public attribution rather than private information.

## Six-size extension

The 10% and 5% production files were added after the frozen four-size release. Every existing benchmark and result file retains its prior bytes. The [extension manifest](extreme-retention-extension-manifest.json) records private-original and public-copy SHA-256 values, host-path redactions, digest-reference updates, and the scientific fields checked for numerical equality. Sanitized source copies without runnable path-default edits are retained under `provenance/archived-code/odds_oneclass/extreme_retention/`. A private workflow name for the focal-metrics JSON was changed in the public copy; its numbers and rankings were preserved. The new supervision manifest and analysis audit record `recovery_enabled` in place of private authorization workflow wording; the active watchdog mode and all numerical supervision facts are preserved. The new watchdog snapshots and logs are public sanitized views of saved evidence. `scripts/verify_extreme_extension.py` verifies both frozen and added file hashes, the exact six-size grid, status counts, and metric bounds. The separate adaptive-batch sensitivity adds 370 two-method cells for AutoEncoder and VAE, changing only the training minibatch to `min(32, fitted normals)` while retaining inference batch 32. Those two PyOD detectors remain normal-only methods; the sensitivity addresses the empty default training loader below 32 fitted normals. Its 740 rows are not part of the 57,120 primary ranking records; unchanged historical rows used by the sensitivity retain their original file hashes. A startup timing sidecar records the brief possible CPU overlap for the first eight adaptive cells. Their 16 method runs all succeeded below 5.1 seconds against the 15-second eligibility limit; original timings are retained and may reflect transient load. No manuscript source, paper PDF, or Overleaf link is included.
