# Benchmark Execution Plan

This document turns the project narrative into non-negotiable benchmark rules. Its purpose is to prevent soft baselines, uneven tuning, and post-hoc reinterpretation of results.

## North-Star Question

Do amplitude, angle, or basis quantum-inspired feature encodings provide a meaningful representational advantage over strong classical alternatives when all methods are compared under matched capacity, matched optimization opportunity, and explicit practical-cost accounting?

## Non-Negotiable Principles

### 1. Fair Comparison

Every major comparison must satisfy all of the following unless an exception is opened as a documented decision record:

- Matched output feature dimensionality or explicitly matched representational budget.
- Matched optimization budget.
- Matched regularization family and comparable regularization strength search.
- Matched tuning effort across QIE and classical baselines.
- Matched seed count and evaluation protocol.

The project may not claim advantage from a setup where QIE receives more features, more training time, more hyperparameter search, or more forgiving regularization than its comparators.

### 2. Strong Baseline Set

The baseline suite is locked to include:

- Scaled linear baseline.
- RBF kernel baseline.
- Polynomial kernel baseline.
- Random Fourier Features baseline.
- Learned embedding baseline, such as an autoencoder-derived representation.
- A simple but competent MLP baseline that is not intentionally underpowered or under-tuned.

If a baseline is dropped for a given dataset, the reason must be logged in the corresponding experiment record and phase record.

### 3. Dataset Requirement

The benchmark suite includes ten datasets across six categories.  This roster
was expanded from the original four-category requirement during Phase 1 review
to meet publication-quality standards.

| Dataset | Category | Samples | Features | Classes | Reason for inclusion |
|---|---|---|---|---|---|
| UCI Wine | Tabular | 178 | 13 | 3 | Baseline tabular benchmark |
| UCI Breast Cancer | Tabular | 569 | 30 | 2 | Second tabular; generalises Wine findings |
| Dry Bean | Tabular | 13,611 | 16 | 7 | Harder multiclass tabular problem |
| Credit Card Fraud | Financial | 284,807 | 30 | 2 | Real financial data; PCA-preprocessed inputs |
| Fashion-MNIST | Image | 70,000 | 784 | 10 | Primary image benchmark |
| CIFAR-10 (flat) | Image | 60,000 | 3,072 | 10 | Harder image benchmark; expected by reviewers |
| HIGGS (500k) | Physics | 500,000 | 28 | 2 | Canonical QML benchmark |
| High-dim parity | Synthetic | 10,000 | 20 | 2 | Controlled synthetic; known decision boundary |
| High-rank noise | Stress-test | 5,000 | 200 | 2 | Representational stress-test |
| Covertype | Large-scale | 581,012 | 54 | 7 | Tests overhead at scale |

The stress-test dataset (high-rank noise) cannot be included merely as
decoration.  It must appear in the main comparison narrative and not only
in a supplemental appendix.

### 4. Explanation, Not Just Scorekeeping

Accuracy and F1 are necessary but insufficient. The analysis plan must include, where scientifically relevant:

- NTK analysis.
- CKA or equivalent representation-similarity analysis.
- Effective rank.
- Condition number or related conditioning diagnostics.
- Spectral-decay analysis.
- Convergence and stability behavior.
- Memory cost and wall-clock cost.

Any use of NTK or CKA must answer a concrete question, such as whether QIE induces meaningfully different geometry from classical kernels or whether observed gains are reducible to known feature-lifting behavior.

### 5. Practical Overhead Is Part of the Result

Encoding overhead is not an implementation footnote. It is part of the benchmark outcome.

- Encoding wall-clock time must be reported separately from downstream model-training time.
- Memory overhead must be recorded where encodings inflate representation size materially.
- If encoding cost is comparable to or greater than the useful training cost, that must be highlighted in the main discussion.
- A method that improves a metric slightly but imposes disproportionate encoding cost cannot be presented as practically favorable without qualification.

## Phase-Specific Requirements

### Phase 0: Scope Lock

Freeze:

- Encoding set: amplitude, angle, basis.
- Baseline set listed above.
- Capacity-matching rule.
- Tuning-equality rule.
- Practical-overhead reporting rule.
- Primary interpretation outcomes: geometric advantage, numerical equivalence, or negative result.

### Phase 1: Infrastructure and Determinism

Before broad experimentation:

- Confirm deterministic reruns under fixed seeds.
- Confirm configs can recreate one full comparison path end to end.
- Confirm profiling hooks capture encoding time, training time, and memory use.

### Phase 2: Numerical Behavior Before Training

For each encoding:

- Verify normalization and dimensional consistency.
- Measure feature-matrix spectra.
- Measure effective rank and conditioning.
- Test sensitivity to noise and input scaling where appropriate.

No downstream performance claim should appear before this stage is documented.

### Phase 3: Dataset Freeze

Freeze the final dataset roster and explicitly justify:

- Why each dataset is scientifically informative.
- Why at least one dataset stresses representation quality rather than trivial pattern recognition.
- Why any discarded dataset was removed.

### Phase 4: Full Sweep

For each dataset and method family:

- Run the same seed count.
- Use the same train, validation, and test protocol.
- Use matched tuning effort and document the search space.
- Record failures, reruns, and excluded results.

### Phase 5: Attribution Analysis

This phase must answer:

- Are QIE representations geometrically distinct from classical baselines using `cka_analysis.py`?
- If performance differs, is the difference associated with rank expansion (e-rank), conditioning (kappa), spectral behavior, or optimization geometry (NTK)?
- Are any gains worth the practical encoding overhead?

Outcome validation:
- Run CKA similarity matrices across all (encoding, baseline) pairs for each dataset to distinguish 'unique' representations (CKA < 0.6) from 'redundant' ones (CKA > 0.9).

## Explicit Stop Rules

The team must halt and open a decision record if any of the following occurs:

- Reproducibility fails under fixed seeds.
- Capacity matching becomes ambiguous for a claimed comparison.
- One method family receives materially more tuning effort than the others.
- The hard stress-test dataset is removed without replacement.
- Encoding overhead dominates the workflow enough that practical claims would become misleading.

## What Counts As A Convincing Result

### Strong Positive Result

QIE shows a reproducible benefit over strong baselines under matched budgets, and the benefit is supported by geometric or optimization analysis rather than just endpoint accuracy.

### Strong Equivalence Result

QIE performs similarly to classical baselines and NTK, CKA, rank, or spectral evidence shows that the methods induce comparable geometry under matched budget. This is publishable because it clarifies what QIE is and is not buying.

### Strong Negative Result

QIE underperforms or only matches classical baselines while costing materially more in time or memory. This is publishable if documented rigorously and framed as a practical and theoretical constraint.

## Anti-Patterns

The paper should avoid:

- Claiming novelty from weak baselines.
- Treating MNIST-like wins as decisive by default.
- Reporting only accuracy and omitting representational diagnostics.
- Hiding overhead in implementation details.
- Changing benchmark rules after seeing results.
- Using NTK or CKA figures that do not support a specific interpretive claim.
