# [PHASE 0] Scope Lock And Narrative Freeze

- Record type: phase-outcome
- Phase: 0 - Scope lock
- Status: Locked
- Owner: Fluxx
- Related branch or Pull Request: `week_0_narrative_lock_in`

## Objective

Freeze the project scope, narrative frame, baseline set, fairness rules, and measurement criteria before Week 1 infrastructure work begins. Phase 0 establishes what the paper is allowed to claim and what evidence is required before any claim can be made.

## Frozen Project Framing

This project investigates whether quantum-inspired feature encodings (QIE), specifically amplitude, angle, and basis encodings, provide a unique geometric advantage when used as explicit feature maps inside classical machine learning pipelines.

The central question is not whether QIE can work. The central question is whether any observed benefit is geometrically distinctive, or whether the same behavior is reproducible with strong classical nonlinear representations such as kernels, random features, learned embeddings, or a competent MLP baseline.

QIE is treated as a nonlinear feature-map family:

```text
x -> phi(x) -> f(phi(x))
```

The project does not claim quantum computation or hardware-dependent quantum advantage. It tests whether the induced feature-space geometry has measurable properties that are not achievable by classical baselines under equal constraints.

## Accepted Outcome Space

The final paper must accept exactly one of these evidence-based directions:

- Geometric advantage: QIE outperforms classical baselines and the difference is supported by measurable rank, conditioning, spectral, NTK, or representation-similarity evidence.
- Numerical equivalence: QIE representations are geometrically or spectrally equivalent to classical baselines under matched budget.
- Negative result: QIE underperforms or only matches classical baselines while imposing unnecessary cost.

Equivalence and negative results are valid endpoints, not fallback narratives.

## Frozen Claim Boundaries

Primary claim:

QIE is a nonlinear feature-map family whose behavior should be explained through induced geometry, spectral structure, rank expansion, conditioning, NTK behavior, and representation similarity rather than through vague quantum expressivity.

Secondary claim:

Any optimization benefit should be attributed to conditioning, NTK behavior, or representational geometry, not to hardware-dependent quantum advantage.

## Frozen Encoding Set

- Amplitude encoding.
- Angle encoding.
- Basis encoding.

No new encodings may be added after Phase 0 unless a decision record explicitly reopens scope.

## Frozen Baseline Set

All QIE results must be interpreted relative to strong classical alternatives:

- Scaled linear baseline.
- RBF kernel baseline with tuned bandwidth.
- Polynomial kernel baseline with degree 2 and degree 3 variants.
- Random Fourier Features baseline.
- Learned embedding baseline, such as an autoencoder-derived representation.
- Simple but competent MLP baseline that is not intentionally underpowered or under-tuned.

Baselines are not treated as trivial preprocessing. They are the reference frame for testing whether QIE produces geometry that cannot be replicated by classical nonlinear mappings.

## Dataset Envelope

The benchmark suite includes the following ten datasets organised by category.
This roster was expanded from the original four-category envelope during Phase 1
review to meet the standard expected of a publication-quality benchmark.  The
expansion is documented here as a scope amendment; no encoding, baseline, or
fairness rule was changed.

### Tabular

- **UCI Wine** — 178 samples, 13 features, 3 classes.  Baseline tabular
  benchmark; used in Phase 1 and Phase 2 infrastructure work.
- **UCI Breast Cancer** — 569 samples, 30 features, binary.  Second tabular
  benchmark; increases confidence that tabular findings generalise beyond Wine.
- **Dry Bean** — 13,611 samples, 16 features, 7 classes.  Harder multiclass
  tabular problem; tests QIE geometric properties under more classes and larger
  sample counts than Wine or Breast Cancer alone.

### Financial

- **Credit Card Fraud Detection** — 284,807 samples, 30 PCA-transformed
  features, binary (imbalanced).  Real financial data with clean public
  provenance.  The PCA preprocessing tests whether QIE adds value on top of
  an existing linear projection, which is directly relevant to the central
  geometric question.

### Image

- **Fashion-MNIST** — 70,000 samples, 784 features, 10 classes.  Satisfies
  the image category requirement and is preferred over plain MNIST.
- **CIFAR-10 (flattened)** — 60,000 samples, 3,072 features, 10 classes.
  Fashion-MNIST is considered easy by current standards.  CIFAR-10 has richer
  texture and colour structure and is expected by reviewers at top venues.

### Physics / Large-Scale

- **HIGGS (500k subset)** — 500,000 samples, 28 features, binary.  Canonical
  large-scale benchmark in the quantum ML literature.  Reviewers familiar with
  QML papers will specifically expect this dataset.

### Synthetic

- **High-dimensional parity** — 10,000 samples, 20 features, binary.  Provably
  hard for linear models.  Label is determined by the parity of the signs of
  the first 5 features, giving a known decision boundary for controlled analysis.
- **High-rank noise** — 5,000 samples, 200 features, binary.  Low-rank signal
  embedded in high-rank isotropic noise.  Directly tests whether encodings that
  collapse to low-dimensional representations fail under representational pressure.

### Large-Scale Tabular

- **Covertype** — 581,012 samples, 54 features, 7 classes.  Tests whether
  encoding overhead remains practical at large scale and whether geometric
  properties observed on small datasets hold as sample count grows.

### Pruning rules

- Drop any dataset where a linear model exceeds 95 percent performance unless
  retained only as a smoke test.
- Drop any dataset where no method beats random chance plus a documented tolerance.
- The high-rank or high-intrinsic-dimensional stress test cannot be removed
  without replacement and a decision record.
- No new datasets may be added after Week 3.

## Frozen Fairness Rules

All headline comparisons must satisfy:

- Matched output feature dimensionality, written as `d_out`.
- Matched optimization budget.
- Matched regularization policy and comparable regularization search.
- Matched tuning effort across QIE and classical baselines.
- Matched seed count and evaluation protocol.
- Explicit accounting for encoding wall-clock time and memory overhead.

Only one scientific variable should change in a controlled comparison: the encoding.

## Measurement Framework

Performance:

- Accuracy.
- F1-score.
- Confidence intervals across independent seeds.
- Paired statistical tests where comparisons are central to a claim.

Representation geometry:

- Singular value distribution.
- Effective rank, written as `erank`.
- Spectral decay.
- Condition number, written as `kappa`.
- Linear probe accuracy.

Representation similarity and optimization behavior:

- Centered Kernel Alignment between QIE Gram matrices and classical kernel Gram matrices.
- NTK comparison across QIE and classical baselines.
- Training loss convergence rate.
- Gradient norm behavior.
- Stability across seeds.
- Failure modes, NaNs, and reruns.

Efficiency:

- Encoding wall-clock time.
- Training wall-clock time.
- Total pipeline wall-clock time.
- Peak memory footprint where practical.
- Feature dimensionality and storage footprint.

## Week 0 Findings

Conceptual findings:

- QIE is best understood as a structured nonlinear feature transformation, not as quantum computation.
- The relevant object of study is the geometry of the encoded feature matrix.
- Potential gains must be explained through rank, conditioning, spectra, NTK behavior, or representation similarity.

Classical comparator findings:

- RBF kernels provide a strong smooth nonlinear reference.
- Polynomial kernels provide a controlled interaction-degree reference.
- Random Fourier Features provide a direct finite-dimensional explicit-map competitor.
- Scaled linear models provide a no-lifting control.
- Learned embeddings and MLPs prevent the study from relying only on weak or overly narrow baselines.

Geometry findings:

- Singular value spectra reveal whether encoded features use many directions or collapse into low-dimensional structure.
- High effective rank suggests broader use of representational capacity, but it is not sufficient by itself to prove improved generalization.
- High condition number indicates numerical instability and may explain poor optimization.
- Better optimization must be attributed to conditioning or NTK behavior when the evidence supports that attribution.

Representation-equivalence findings:

- CKA is the primary tool for testing whether QIE and classical kernel representations induce similar geometry.
- CKA near 1.0 would support an equivalence interpretation rather than a unique QIE advantage.
- CKA substantially below 1.0 would indicate distinct geometry, but predictive and efficiency evidence would still be required before claiming practical advantage.

Failure modes identified:

- Unequal feature dimensionality.
- Unequal training budgets.
- Unequal regularization policies or tuning effort.
- Architecture changes after the architecture freeze.
- Weak baseline selection.
- Attributing gains to quantum effects instead of measured geometry.
- Ignoring encoding overhead, memory cost, spectral behavior, or stability failures.

## Operational Readiness For Week 1

Week 0 establishes the conceptual, analytical, and experimental constraints needed for Week 1 infrastructure and determinism work:

- Problem space is defined.
- Accepted outcome space is fixed.
- Encodings and baselines are identified.
- Metrics and diagnostic families are defined.
- Invalid comparison patterns are documented.
- The Week 1 implementation target is clear: build deterministic, config-driven infrastructure that preserves these constraints.

## Required Evidence And Artifact Paths

- Frozen title: [README.md](../../README.md) and [EXECUTION_PLAN.md](../../EXECUTION_PLAN.md).
- Frozen abstract and project framing: [README.md](../../README.md).
- Frozen objective and research contract: [EXECUTION_PLAN.md](../../EXECUTION_PLAN.md).
- Frozen encoding set: [EXECUTION_PLAN.md](../../EXECUTION_PLAN.md).
- Frozen baseline set: [EXECUTION_PLAN.md](../../EXECUTION_PLAN.md).
- Dataset envelope: [EXECUTION_PLAN.md](../../EXECUTION_PLAN.md).
- Matched-budget and matched-tuning rules: [EXECUTION_PLAN.md](../../EXECUTION_PLAN.md) and [docs/benchmark_execution_plan.md](../../docs/benchmark_execution_plan.md).
- Branch-protected workflow rules: [CONTRIBUTING.md](../../CONTRIBUTING.md) and [docs/governance/peer_review_protocol.md](../../docs/governance/peer_review_protocol.md).
- Phase verification standard: [docs/governance/phase_outcome_matrix.md](../../docs/governance/phase_outcome_matrix.md).
- Repository-record workflow: [records/README.md](../README.md).

## Verification Method

One teammate should verify that:

- The encoding list is frozen to amplitude, angle, and basis encodings.
- The baseline list includes scaled linear, RBF, polynomial, Random Fourier Features, learned embedding, and non-handicapped MLP comparators.
- The dataset envelope includes non-trivial tabular, image, synthetic high-dimensional, and high-rank or high-intrinsic-dimensional stress-test categories.
- Matched-budget, matched-optimization, matched-regularization, matched-tuning, and overhead-accounting rules are documented.
- No unapproved scope creep remains open.

After verification, the reviewer may change `Status` from `Ready for review` to `Locked`.

## Outcome Status

Ready for Week 1 review. The target final status is `Locked` after teammate verification.

## Outcome Narrative

Not yet determined. The project remains pre-result and explicitly allows geometric advantage, numerical equivalence, or negative result.

## Completion Checklist

- [x] Frozen title is recorded.
- [x] Frozen abstract and project framing are recorded.
- [x] Dataset envelope is recorded.
- [x] Encoding set is frozen.
- [x] Baseline set is frozen and includes a non-handicapped MLP comparator.
- [x] Matched-budget rules are recorded.
- [x] Matched-tuning rules are recorded.
- [x] Branch-protected workflow rules are recorded in governance docs.
- [x] Final artifact paths are listed.
- [x] Related Pull Request or final commit range is listed.

## Closure Statement

This signifies the completion of phase 0, arguably the easiest phase we'll encounter
