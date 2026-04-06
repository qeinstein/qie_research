# Data Directory

This project separates tracked provenance from untracked heavy binaries.

- `raw/` holds immutable source-aligned inputs but is ignored by Git except for placeholders.
- `processed/` holds model-ready derivatives and is also ignored by Git unless a team decision explicitly promotes a tiny example artifact.
- `metadata/` is the canonical tracked layer for provenance, schema notes, checksums, licenses, and collection history.

