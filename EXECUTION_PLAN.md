# Research Execution Plan: Quantum-Inspired Feature Encodings

Project title: Benchmarking Quantum-Inspired Feature Encodings as Explicit High-Dimensional Representations in Classical Machine Learning

## North-Star Objective

Determine whether quantum-inspired feature encodings (QIE) provide a unique geometric advantage, measured through rank expansion, spectral decay, conditioning, NTK behavior, and representation similarity, that is not achievable under an equal representational budget by strong classical baselines.

In this project, "not achievable" means:

- Matched output feature dimensionality, written as `d_out`.
- Matched optimization budget.
- Matched regularization policy and comparable regularization search.
- Matched tuning effort across QIE and classical baselines.
- Explicit accounting for encoding wall-clock time and memory overhead.

## Research Contract

The paper must accept exactly one of the following final directions based on the evidence:

- Geometric advantage: QIE outperforms classical baselines and the difference is supported by measurable rank, conditioning, spectral, NTK, or representation-similarity evidence.
- Numerical equivalence: QIE representations are geometrically or spectrally equivalent to classical baselines under matched budget. This becomes the equivalence paper.
- Negative result: QIE underperforms or only matches classical baselines while imposing unnecessary cost. This becomes the debunking or practical-limitation paper.

Equivalence and negative results are valid endpoints, not fallback narratives.

## Locked Methods

### Quantum-Inspired Encodings

- Amplitude encoding.
- Angle encoding.
- Basis encoding.

No new encodings are added after Phase 0 unless a decision record explicitly reopens scope.

### Classical Baselines

- Scaled linear baseline.
- RBF kernel baseline with tuned bandwidth.
- Polynomial kernel baseline with degree 2 and degree 3 variants.
- Random Fourier Features baseline.
- Learned embedding baseline, such as an autoencoder-derived representation.
- A simple but competent MLP baseline that is not intentionally underpowered or under-tuned.

## Dataset Plan

The benchmark suite must include:

- Non-trivial tabular data, such as UCI Wine or Breast Cancer.
- Image data, with Fashion-MNIST preferred over easy MNIST-only claims.
- A synthetic high-dimensional dataset such as high-dimensional parity or XOR.
- At least one stress-test dataset with high-rank noise or high intrinsic dimensionality.
- Optional financial or other high-dimensional structured data only if provenance and reproducibility are clean.

Pruning rules:

- Drop datasets where a linear model exceeds 95 percent performance unless the dataset is retained only as a smoke test.
- Drop datasets where no method beats random chance plus a documented tolerance.
- Do not remove the high-rank or high-intrinsic-dimensional stress test without replacing it and opening a decision record.

## Required Metrics

### Predictive Performance

- Accuracy.
- F1-score.
- Confidence intervals across independent seeds.
- Paired statistical tests where comparisons are central to a claim.

### Representation Geometry

- Effective rank, written as `erank`.
- Singular value distribution.
- Spectral decay.
- Condition number, written as `kappa`.
- Linear probe accuracy.
- Centered Kernel Alignment between QIE Gram matrices and classical kernel Gram matrices.

### NTK Analysis

NTK analysis is mandatory because this paper should explain how the encoding changes the optimization geometry, not only whether endpoint accuracy changes.

Required NTK questions:

- Does a QIE representation induce a meaningfully different NTK from the strongest classical baselines?
- Are observed gains associated with NTK conditioning or spectral changes?
- If CKA between QIE and classical kernel representations is near 1.0, are QIE gains actually just classical feature lifting under another parameterization?

### Optimization Behavior

- Training loss convergence rate.
- Stability across seeds.
- Gradient norm behavior.
- Failure modes, NaNs, and reruns.

### Classical Overhead

The encoding cost must be measured separately from downstream model training.

Required overhead metrics:

- Encoding wall-clock time.
- Training wall-clock time.
- Total pipeline wall-clock time.
- Peak memory footprint where practical.
- Feature dimensionality and storage footprint.
- Backpropagation overhead for differentiable encodings.

If encoding takes longer than the useful training stage or dominates total runtime, the method cannot be presented as practically favorable without strong qualification.

## Eight-Week Sprint

### Week 0: Scope Lock and Narrative Freeze

Primary claim:

QIE is a nonlinear feature-map family whose behavior should be explainable through induced geometry and rank expansion rather than hardware-dependent quantum advantage.

Secondary claim:

Any optimization benefit should be attributed to conditioning, NTK behavior, or representational geometry, not vaguely to quantum expressivity.

Hard rules:

- No new encodings after Week 0.
- No new datasets after Week 3.
- No architecture changes after Week 2.
- No post-hoc weakening of baselines after results are observed.

Deliverables:

- Frozen encoding list.
- Frozen baseline list.
- Frozen fairness rules.
- Repository phase record for Phase 0 closure at `records/phases/phase-0-scope-lock.md`.

### Week 1: Infrastructure and Determinism

Tasks:

- Build a config-driven runner.
- Enforce strict seed control.
- Add logging for loss, F1, gradient norm, wall-clock time, memory where practical, and artifact paths.
- Ensure single-command execution for one model, one dataset, and all encodings.

Deliverable:

- One deterministic smoke path that can be rerun by another teammate.

Stop condition:

- Halt if 100 percent reproducibility is not achieved for the designated deterministic smoke run.

### Week 2: Numerical Behavior Before Training

Tasks:

- Verify amplitude normalization: sum_i |c_i|^2 = 1.
- Verify shape and dimensionality of each encoding.
- Compute singular value distributions of the encoded feature matrix.
- Compute effective rank and condition number.
- Test stability under noise, for example epsilon ~ N(0, 0.01).

Deliverable:

- Internal encoding behavior report.

Rule:

- Do not make performance claims before this numerical behavior audit is complete.

### Week 3: Dataset Expansion and Stress Testing

Final dataset categories:

- Tabular benchmark.
- Image benchmark.
- Synthetic high-dimensional parity, XOR, or related stress benchmark.
- High-rank-noise or high-intrinsic-dimensional dataset.

Deliverables:

- Frozen dataset registry.
- Provenance and checksum records.
- Dataset pruning decisions with reasons.

### Week 4: Full Experimental Sweep

Design constraints:

- Matched `d_out`.
- Matched optimization budget.
- Matched regularization policy.
- Matched tuning effort.
- Five independent seeds unless compute constraints are documented.
- Fixed training-budget ceiling, with deviations logged.

Deliverables:

- Raw result summary tables.
- Runtime and memory overhead summaries.
- Draft descriptive figures.
- Failure and rerun log.

### Week 5: Analysis and the "Why"

Mandatory analyses:

- Centered Kernel Alignment between QIE and classical kernel representations.
- NTK comparison across QIE and classical baselines.
- Effective-rank versus performance analysis.
- Conditioning versus convergence analysis.
- Spectral-decay comparison.
- Classical overhead versus performance trade-off.

Mandatory ablation:

- Compare high-dimensional amplitude encoding against a Random Fourier Features baseline with matched output dimensionality.

Deliverables:

- Two to three core figures.
- Primary figure linking effective rank or conditioning to performance.
- Practical overhead table.
- Written interpretation choosing geometric advantage, numerical equivalence, or negative result as the leading direction.

### Week 6: Draft v1

Tasks:

- Draft the paper in a skeptical tone.
- Frame QIE as a geometric representation tool, not as a quantum-advantage claim.
- Link every major claim to a result artifact.

Deliverable:

- Full draft v1.

### Week 7: Reviewer Hardening

Tasks:

- Stress-test claims against likely reviewer objections.
- Add complexity analysis.
- Add appendix material, including Wirtinger calculus details if complex-valued gradients are used.
- Check that overhead claims are visible in the main text.

Deliverable:

- Reviewer-hardening notes and revised draft.

### Week 8: Figure Polish and Submission Package

Tasks:

- Export publication-quality vector figures where possible.
- Freeze final tables.
- Clean repository records and Pull Requests.
- Confirm reference library consistency.
- Prepare submission or arXiv package.

Deliverable:

- Submission-ready manuscript package.

## Technical Notes

### Centered Kernel Alignment

CKA measures similarity between Gram matrices from different feature representations. It matters because it is invariant to orthogonal transformations and isotropic scaling. If QIE is effectively the same as an RBF or other classical kernel under transformation, CKA should expose that similarity.

### Wirtinger Calculus

Complex-valued mappings may not be holomorphic because typical ML losses depend on quantities such as |z|^2. If complex-valued encodings are optimized directly, gradients should be handled with the appropriate complex-gradient logic, such as treating z and its conjugate as independent variables where needed.

### Effective Rank and Spectral Decay

Effective rank uses the entropy of normalized singular values to quantify how evenly a representation uses its feature dimensions. High effective rank may indicate broader use of representational capacity, while strong spectral decay may indicate low-dimensional structure or implicit regularization.

### Representational Budget

All headline comparisons must control representational budget. Claims are invalid if QIE receives substantially more output features, parameters, tuning attempts, or compute budget than the baseline it is said to beat.

## Final Principle

The goal is not to defend QIE. The goal is to measure it honestly.
