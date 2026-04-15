# [PHASE 2] Numerical Behaviour Before Training

- Record type: phase-outcome
- Phase: 2 - Numerical behaviour before training
- Status: Ready for review
- Owner: Fluxx
- Related branch or Pull Request: `phase-1-implementation`

## Objective

Audit the numerical behaviour of all three encodings on real data before
any training or performance claims are made.  Verify structural invariants,
characterise the geometry of each encoded feature matrix, and measure
stability under noise.

## Deliverables

### 1. Numerical audit script

[`src/qie_research/analysis/numerical_audit.py`](../../src/qie_research/analysis/numerical_audit.py)

Runs all checks across all three encodings on UCI Wine and writes results
to [`results/metrics/phase_2_numerical_audit.json`](../../results/metrics/phase_2_numerical_audit.json).

Invoked as:

```bash
python -m qie_research.analysis.numerical_audit
```

### 2. LaTeX report

[`docs/reports/phase_2_numerical_audit.tex`](../../docs/reports/phase_2_numerical_audit.tex)

Internal audit report covering invariant verification, singular value
distributions, effective rank, condition number, and noise stability.
Compiled PDF to be added to the same directory by the author.

### 3. Raw audit output

[`results/metrics/phase_2_numerical_audit.json`](../../results/metrics/phase_2_numerical_audit.json)

## Audit Results

Dataset: UCI Wine, 142 training samples, 13 features, seed 42.

| Encoding | Invariant | erank | kappa | Noise delta |
|---|---|---|---|---|
| amplitude | PASS | 1.38 | 6650.85 | 5.0e-6 |
| angle | PASS | 23.28 | 7.55 | 1.4e-2 |
| basis | PASS | 77.09 | 108.73 | 2.9e-1 |

## Key Findings

**Amplitude encoding** is nearly rank-1 and severely ill-conditioned
(kappa = 6650.85). L2 normalisation collapses the feature space to a near
one-dimensional structure. This must be reconciled with Phase 4 performance
results and explained through conditioning and NTK analysis in Phase 5.

**Angle encoding** achieves near-full effective rank utilisation (23.28 out
of 26, 89.6%) with a low condition number (7.55). It is the most
geometrically well-behaved encoding under these conditions.

**Basis encoding** has the highest absolute effective rank (77.09) but is
the most sensitive to noise (delta = 0.29) due to the discontinuous
quantisation step. Bit flips near bin boundaries amplify small perturbations
discontinuously. This is a structural property that must be reported in the
Phase 4 practical overhead discussion.

## Stop Rule Compliance

No performance claims appear in this report. No downstream model results
are cited. All numbers are pre-training geometric diagnostics only.

## Verification Method

A teammate should verify that:

- `python -m qie_research.analysis.numerical_audit` runs without error and
  produces [`results/metrics/phase_2_numerical_audit.json`](../../results/metrics/phase_2_numerical_audit.json).
- All three invariant checks show `"passed": true` in the JSON output.
- The LaTeX report compiles cleanly on Overleaf.
- No accuracy, F1, or model performance numbers appear anywhere in the report.

After verification, the reviewer may change `Status` from `Ready for review`
to `Locked`.

## Artifact Paths

- Audit script: [`src/qie_research/analysis/numerical_audit.py`](../../src/qie_research/analysis/numerical_audit.py)
- Raw output: [`results/metrics/phase_2_numerical_audit.json`](../../results/metrics/phase_2_numerical_audit.json)
- LaTeX report: [`docs/reports/phase_2_numerical_audit.tex`](../../docs/reports/phase_2_numerical_audit.tex)

## Completion Checklist

- [x] Invariant checks pass for all three encodings.
- [x] Singular value distributions computed and reported.
- [x] Effective rank computed and reported.
- [x] Condition number computed and reported.
- [x] Noise stability measured under epsilon ~ N(0, 0.01).
- [x] LaTeX report written with all findings.
- [x] No performance claims made anywhere in this phase.

## Closure Statement

Phase 2 numerical audit is complete. All encodings are numerically
well-defined and structurally sound. Phase 3 dataset freeze may begin.
