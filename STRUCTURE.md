# Repository Structure

This is the final first-paper structure. It keeps the folders expected to be useful for this project so the team does not need to add new folders later.

## Tree

```text
.
|-- .github/                  Pull Request templates.
|-- configs/                  Experiment and run configuration files.
|-- data/                     Dataset storage and dataset provenance.
|   |-- raw/                  Original source-aligned data. Large files are ignored.
|   |-- processed/            Model-ready data generated from code and configs. Large files are ignored.
|   |-- metadata/             Tracked dataset notes, provenance, checksums, and schemas.
|-- docs/                     Project plans and governance documents.
|   |-- governance/           Logging, commit, review, reference, phase protocols, and record templates.
|-- manuscript/               Drafts, LaTeX, figures, tables, and submission material as they appear.
|-- references/               Bibliography, reference PDFs, and reading notes.
|   |-- bibtex/               Canonical BibTeX file.
|   |-- pdf_library/          Reference PDFs when legal and practical to store.
|   |-- reading_notes/        Notes keyed by citation or paper topic.
|-- results/                  Reviewable outputs from experiments and analysis.
|   |-- figures/              Figures for drafts, reports, and the final paper.
|   |-- tables/               Result tables.
|   |-- metrics/              Metric summaries, NTK summaries, statistical tests, and compact CSV outputs.
|   |-- logs/                 Runtime, profiling, and evaluation logs. Large logs are ignored.
|-- src/                      Project source code.
|   |-- qie_research/         Importable Python package.
|       |-- analysis/          Benchmark analysis, NTK analysis, representation metrics, and statistics.
|       |-- baselines/         Classical baselines such as RBF, polynomial, RFF, learned embeddings, and MLPs.
|       |-- encodings/         Amplitude, angle, and basis encoding implementations.
|       |-- pipelines/         Data, training, evaluation, and reporting workflows.
|       |-- utils/             Shared helpers.
|-- tests/                    Tests and small fixtures.
|-- records/                  Repository-native records for phases, experiments, and decisions.
|   |-- phases/               Formal phase outcome records.
|   |-- experiments/          Experiment registration and run-family records.
|   |-- decisions/            Scope, method, dataset, narrative, or tooling decisions.
```

## Root Files

- `README.md`: project overview, abstract, and operating rules.
- `EXECUTION_PLAN.md`: the main eight-week execution plan.
- `CONTRIBUTING.md`: Git workflow and review rules.
- `STRUCTURE.md`: this folder map.
- `.gitignore`: prevents large data, secrets, caches, and build artifacts from entering Git.

## Rules

- Keep this tree stable unless a strong project requirement forces a change.
- Do not add new top-level folders without a clear reason.
- Put general plans and process documents in `docs/`.
- Put all reusable scientific code under `src/qie_research/`.
- Put raw data, processed data, and generated logs in purpose-specific folders when they exist, but keep large files out of Git.
- Put compact, paper-relevant outputs in `results/`.
- Put manuscript source and draft material in `manuscript/`.
- Put operational records in `records/`.
