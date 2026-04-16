# Matched-Budget Benchmark Rules (Phase 4)

This document specifies the exact constraints enforced during the Phase 4
Experimental Sweep. These rules ensure that any observed performance
differences are attributable to the **representation geometry** of the
encoding, rather than differences in optimization, tuning effort, or
representational capacity.

## 1. Representational Budget (d_out)

*   **Rule:** Every classical baseline using an explicit feature map (RFF, PCA, Polynomial) must match the **median output dimensionality** of the QIE encodings for that specific dataset.
*   **Implementation:** Calculated dynamically in `runner.py`. For a 13-feature input (Wine), this typically results in a matched `d_out` of **49** (median of Amplitude=16, Angle=26, Basis=104).
*   **Justification:** Prevents Basis Encoding from inflating the baseline budget while ensuring Amplitude Encoding is not compared against an unfairly low-dimensional baseline.

## 2. Optimization Budget (Linear Heads)

*   **Rule:** All linear classification heads (Logistic Regression) must use identical hyperparameters across QIE and baselines.
*   **Hyperparameters:**
    *   `C = 1.0` (Inverse regularization strength)
    *   `max_iter = 1000` (Convergence ceiling)
    *   `solver = 'lbfgs'` (Default sklearn solver)
*   **Consistency:** The same `C` and `max_iter` apply to `raw_linear`, `rff`, `poly`, `pca`, and the linear head following `amplitude`, `angle`, and `basis`.

## 3. Neural Network Fairness (MLP)

*   **Rule:** The "Classical MLP" baseline and the "Torch MLP" end-to-end model must use the exact same architecture.
*   **Architecture:** `[256, 128]` hidden layers with ReLU activation.
*   **Optimizer (Torch Path):**
    *   `Learning Rate: 1e-3`
    *   `Weight Decay: 1e-4`
    *   `Batch Size: 256`
    *   `Epochs: 100`
*   **Consistency:** These settings are shared between the `torch_mlp` baseline and the `torch_linear_head` used to analyze QIE convergence curves.

## 4. Evaluation Protocol

*   **Seeds:** Every (dataset, method) pair is executed 5 times using the seeds defined in `configs/seed_registry.yaml`.
*   **Splits:** `test_size = 0.2` (80/20 train/test split) using stratified sampling to preserve class balance.
*   **Standardization:**
    *   `AngleEncoding` uses internal min-max scaling to `[-1, 1]`.
    *   `BasisEncoding` uses internal min-max scaling to `[0, 1]`.
    *   All classical baselines (RFF, Poly, PCA, MLP) use `StandardScaler` (Z-score) to ensure numerical stability and fair gradient flow.

## 5. Large-Scale Handling (max_samples)

*   **Rule:** On datasets exceeding 100,000 samples (HIGGS, Covertype, Credit Card Fraud), expensive methods like **RBF-SVM** may use a `max_samples` constraint.
*   **Fairness:** If `max_samples` is applied to a baseline, it MUST be applied to the QIE training stage for that same dataset. No method is allowed to "see" more data than its competitors.

## 6. Metric Integrity

*   **Primary Metrics:** Accuracy and F1-macro.
*   **Reporting:** All Phase 4 tables must report the **Mean ± Standard Deviation** across the 5 seeds. A result is only considered "superior" if it passes a paired t-test or if the confidence intervals do not overlap.
