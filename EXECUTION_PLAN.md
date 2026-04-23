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

The benchmark suite includes ten datasets across six categories.  This roster
was expanded from the original four-category envelope during Phase 1 review.
The expansion strengthens breadth of evidence and meets the standard expected
of a publication-quality benchmark.  No encoding, baseline, or fairness rule
was changed.

### Tabular

- UCI Wine (178 samples, 13 features, 3 classes) — baseline tabular benchmark.
- UCI Breast Cancer (569 samples, 30 features, binary) — second tabular
  benchmark; increases confidence that tabular findings generalise beyond Wine.
- Dry Bean (13,611 samples, 16 features, 7 classes) — harder multiclass
  tabular problem; tests QIE geometry under more classes and larger sample counts.

### Financial

- Credit Card Fraud Detection (284,807 samples, 30 PCA features, binary,
  imbalanced) — real financial data with clean public provenance; PCA
  preprocessing tests whether QIE adds value on top of an existing linear
  projection.

### Image

- Fashion-MNIST (70,000 samples, 784 features, 10 classes) — satisfies the
  image category requirement; preferred over plain MNIST.
- CIFAR-10 flattened (60,000 samples, 3,072 features, 10 classes) — harder
  image benchmark with richer texture and colour structure; expected by
  reviewers at top venues alongside Fashion-MNIST.

### Physics / Large-Scale

- HIGGS 500k subset (500,000 samples, 21 features, binary) — canonical
  large-scale benchmark in the quantum ML literature; reviewers familiar with
  QML papers will specifically expect this dataset.

### Synthetic

- High-dimensional parity (2,000 samples, 50 features, binary) — provably
  hard for linear models; known decision boundary for controlled analysis.
- High-rank noise (2,000 samples, 100 features, binary) — low-rank signal
  in high-rank isotropic noise; stress-tests encodings that collapse to
  low-dimensional representations.

### Large-Scale Tabular

- Covertype (581,012 samples, 54 features, 7 classes) — tests whether
  encoding overhead remains practical at large scale.

### Pruning rules

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

Final dataset roster (ten datasets across six categories):

- Tabular: UCI Wine, UCI Breast Cancer, Dry Bean.
- Financial: Credit Card Fraud Detection.
- Image: Fashion-MNIST, CIFAR-10 (flattened).
- Physics / Large-Scale: HIGGS (500k subset).
- Synthetic: High-dimensional parity, High-rank noise.
- Large-Scale Tabular: Covertype.

Deliverables:

- Frozen dataset registry.
- Provenance and checksum records.
- Dataset pruning decisions with reasons.
- Data preparation instructions for datasets requiring local caching
  (Fashion-MNIST, CIFAR-10, HIGGS, Covertype, Credit Card Fraud).

### Week 4: Full Experimental Sweep

Design constraints:

- Matched `d_out`.
- Matched optimization budget.
- Matched regularization policy.
- Matched tuning effort.
- Five independent seeds unless compute constraints are documented.
- Fixed training-budget ceiling, with deviations logged.

Deferred from Phase 1:

- Add a differentiable model with an explicit training loop (PyTorch or JAX).
- Log training loss per epoch for all runs.
- Log gradient norm per epoch for all runs.
- Integrate loss curve and gradient norm curve into the runner output JSON.

These were deferred from Phase 1 because the sklearn solver used in the
infrastructure smoke test does not expose per-iteration loss or gradients.
They must be in place before the full sweep begins.

Deliverables:

- Raw result summary tables.
- Runtime and memory overhead summaries.
- Draft descriptive figures.
- Failure and rerun log.
- Loss curves and gradient norm curves for all runs.

### Week 5: Analysis and the "Why"

Mandatory analyses:

- Centered Kernel Alignment (CKA) between QIE and classical kernel representations using `cka_analysis.py`.
- NTK comparison across QIE and classical baselines.
- Effective-rank versus performance analysis.
- Conditioning versus convergence analysis.
- Spectral-decay comparison.
- Classical overhead versus performance trade-off.

Mandatory ablation:

- Compare high-dimensional amplitude encoding against a Random Fourier Features baseline with matched output dimensionality using CKA to quantify geometric overlap.

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
