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
