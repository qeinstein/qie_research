# Semantic Commit Guide

## Commit Format

Use:

```text
<type>(optional-scope): imperative summary
```

Examples:

- `code(encodings): add amplitude normalization checks`
- `text(introduction): tighten claim about classical equivalence`
- `data(metadata): record Fashion-MNIST provenance and checksum`
- `ref(bibtex): add kernel alignment references`

## Allowed Types

| Type | Use For |
| --- | --- |
| `code` | Source code changes affecting analysis, encodings, baselines, pipelines, or utilities |
| `text` | Manuscript text, notes, or prose-driven documents |
| `data` | Data manifests, provenance, checksums, schema notes, or small tracked metadata changes |
| `ref` | BibTeX updates, reference-source notes, reading notes, citation cleanup |
| `fix` | Bug fixes, numerical corrections, reproducibility fixes, or claim-correcting edits |
| `exp` | Experiment registration, run configuration, result collation, or ablation bookkeeping |
| `infra` | Environment, CI, repository structure, tooling, or configuration scaffolding |
| `review` | Addressing review comments, reviewer hardening, or manuscript response edits |
| `docs` | Non-manuscript documentation improvements when `text` would be too broad |

## Rules

- Use the smallest accurate scope.
- One commit should correspond to one logical concern.
- Keep subject lines under about 72 characters when practical.
- Use the imperative mood: `add`, `fix`, `record`, `freeze`, `compare`, `remove`.
- If the change affects scientific interpretation, explain the rationale in the commit body.
- If the commit closes or advances a repository record, mention the record path or record ID in the body.


## Examples By Scenario

```text
code(ntk): add matched-budget kernel trace estimator
exp(stress-tests): register high-rank-noise benchmark grid
fix(training): correct seed leakage in evaluation loop
text(methods): specify classical overhead accounting rule
ref(reading-notes): add QIE and kernel baseline source papers
infra(repo): initialize governance and manuscript scaffold
```

## Anti-Patterns

- `misc updates`
- `final changes`
- `stuff`
- `fix`
- `update manuscript and code and refs`

