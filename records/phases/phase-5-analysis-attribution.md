# Phase 5: Analysis and Attribution

## Core Question
Are quantum-inspired feature encodings (QIE) geometrically distinct from classical baselines, and is any observed performance difference attributable to rank, conditioning, or unique representational structure?

## Status: Planned
This phase begins after the completion of the Phase 4 Experimental Sweep.

## Analysis Strategy

### 1. Geometric Uniqueness (CKA Analysis)
Using `src/qie_research/analysis/cka_analysis.py`, we will compute the Centered Kernel Alignment between:
- Every QIE encoding (Amplitude, Angle, Basis).
- Every relevant classical baseline (RBF, RFF, PCA).

**Target Metrics:**
- **CKA Similarity Score (0.0 to 1.0)**.
- **Redundancy Threshold**: CKA > 0.9 (Indicates QIE is functionally equivalent to the classical baseline).
- **Uniqueness Threshold**: CKA < 0.6 (Indicates QIE represents data in a fundamentally different way).

### 2. Spectral Analysis & Performance Correlation
- Re-run the Phase 2 spectral diagnostics on the final frozen datasets.
- **Direct Attribution**: Correlate **Effective Rank (e-rank)** and **Condition Number (kappa)** directly with **Accuracy** and **F1-score** from Phase 4.
- **Hypothesis**: Higher e-rank in QIE encodings correlates with a statistically significant 'Accuracy Lift' on non-linear synthetic tasks (e.g., parity) compared to classical linear baselines.

### 3. Representation-to-Accuracy Mapping (The "Why")
- **CKA vs. Accuracy Delta**: Plot the CKA similarity (QIE vs. Classical) against the difference in accuracy. 
- **Goal**: If CKA is low (unique representation) AND Accuracy Delta is high (QIE wins), we have found a 'Geometric Advantage.' If CKA is high AND Accuracy is the same, we have 'Numerical Equivalence.'

### 4. Practical Overhead Accounting
- Finalize the table of Encoding Time vs. Training Time.
- Ratio calculation: `Encoding_Time / Total_Time`.
- Rule: If `Ratio > 0.5`, the method must be flagged as "computationally expensive" in the paper.

## Evidence Requirements for Completion
- [ ] CKA similarity heatmap for each dataset.
- [ ] Scatter plot of e-rank vs. accuracy across all methods.
- [ ] Table of peak memory and wall-clock overhead.
- [ ] NTK condition number logs.
- [ ] Final attribution statement (Geometric Advantage, Numerical Equivalence, or Negative Result).
