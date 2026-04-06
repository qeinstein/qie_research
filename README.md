# Benchmarking Quantum-Inspired Feature Encodings as Explicit High-Dimensional Representations in Classical Machine Learning

Research proposal date: January 26, 2026

## Abstract

This project investigates whether quantum-inspired feature encodings (QIE), specifically amplitude, angle, and basis encodings, provide measurable advantages as explicit high-dimensional representations inside fully classical machine learning pipelines. The central goal is to separate representational effects from hardware claims by benchmarking QIE against strong classical baselines under matched architecture, feature-budget, and optimization constraints. In addition to predictive performance, the study evaluates information preservation, spectral structure, effective rank, conditioning, centered kernel alignment, neural tangent kernel behavior, convergence stability, and classical overhead. The benchmark suite is designed to include non-trivial tabular data, image classification, and at least one stress-test dataset with high-rank noise or high intrinsic dimensionality so that any observed gains are tested under realistic representational pressure rather than simple accuracy-only settings.

The repository is organized for a three-person team operating through versioned repository records and reviewed Pull Requests. Project management, experiment logging, manuscript development, citation maintenance, and review decisions are recorded under `records/` and linked from Pull Requests, with tracked artifacts and explicit verification criteria. Large datasets, cached encodings, and heavyweight model artifacts are intentionally excluded from Git history; instead, provenance, checksums, manifests, and summary outputs are versioned so the full workflow remains reproducible and reviewable.

## Research Focus

- Compare amplitude, angle, and basis encodings as explicit differentiable feature maps.
- Benchmark against matched-capacity classical baselines: scaled linear, RBF, polynomial, random Fourier features, and learned embeddings.
- Include a simple but non-handicapped MLP baseline so QIE is not only compared against kernel-style methods.
- Measure not only accuracy and F1, but also NTK behavior, feature-space geometry, conditioning, spectral decay, convergence dynamics, memory cost, and encoding wall-clock overhead.
- Determine whether any gains are uniquely geometric, numerically equivalent to classical kernels, or negative relative to matched baselines.

## Benchmark Fairness Commitments

- Match output feature dimensionality or representational budget across QIE and classical baselines.
- Match optimization budget, regularization family, and tuning effort rather than allowing stronger tuning only for the preferred method.
- Include at least one genuinely difficult stress-test dataset with high-rank noise or high intrinsic dimensionality.
- Treat practical overhead as a first-class outcome: if encoding cost dominates training cost, that finding is reported as a limitation, not hidden in the appendix.
- Require explanation-oriented analysis, not accuracy-only reporting, through NTK, CKA, effective-rank, conditioning, and spectral diagnostics.

## Repository Operating Rules

- `main` is protected and receives changes only through reviewed Pull Requests.
- Every meaningful task starts from a repository record when it affects scope, experiments, decisions, risks, or phase closure, and ends with linked evidence in a Pull Request.
- Manuscript edits, code edits, data-provenance updates, and bibliography changes all follow semantic commit rules.
- Raw and processed data are logged through manifests and metadata, not committed as large binaries.
- Publication-ready figures, tables, and summary metrics are committed; caches, checkpoints, and transient logs are not.

## Core Governance Documents

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md)
- [STRUCTURE.md](STRUCTURE.md)
- [records/README.md](records/README.md)
- [logging_protocol.md](docs/governance/logging_protocol.md)
- [semantic_commit_guide.md](docs/governance/semantic_commit_guide.md)
- [phase_outcome_matrix.md](docs/governance/phase_outcome_matrix.md)
- [peer_review_protocol.md](docs/governance/peer_review_protocol.md)
- [reference_library_maintenance.md](docs/governance/reference_library_maintenance.md)

## Planned Study Stages

1. Scope lock and narrative freeze.
2. Infrastructure and determinism verification.
3. Numerical behavior audit before training.
4. Dataset freeze with explicit high-rank or high-intrinsic-dimensional stress tests.
5. Full matched-budget benchmark sweep.
6. NTK, CKA, rank, conditioning, and overhead analysis.
7. Drafting, reviewer hardening, and submission packaging.

The root execution plan is specified in [EXECUTION_PLAN.md](EXECUTION_PLAN.md). The focused benchmark rules are also maintained in [benchmark_execution_plan.md](docs/benchmark_execution_plan.md).

## Repository Expectations

- Use feature branches with record-linked names when a repository record exists.
- Keep one scientific claim or implementation concern per Pull Request.
- Record final phase outcomes in `records/phases/` before merging summary manuscripts or result tables.
- Treat negative or equivalence outcomes as valid research endpoints rather than fallback narratives.
