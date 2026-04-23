"""
Centered Kernel Alignment (CKA) Analysis
========================================

A robust similarity metric for comparing high-dimensional feature representations.
Used in Phase 5 of the QIE Research project to determine the 'Geometric Uniqueness'
of quantum-inspired encodings compared to classical RBF or RFF baselines.

Mathematical Background
-----------------------
CKA is a similarity measure between two kernel matrices (Gram matrices) K and L.
It is invariant to orthogonal transformations (rotation) and isotropic scaling.
Unlike standard correlation, CKA can compare feature maps with different 
output dimensions (e.g., comparing 16-dim Amplitude vs 104-dim Basis).

    CKA(K, L) = HSIC(K, L) / sqrt( HSIC(K, K) * HSIC(L, L) )

where HSIC (Hilbert-Schmidt Independence Criterion) is the Frobenius inner
product of the centered kernels:

    HSIC(K, L) = vec(H K H) . vec(H L H)

Algorithm Optimization
----------------------
Standard centering via matrix multiplication (H @ K @ H) is O(n^3).
This implementation uses a vectorized O(n^2) centering algorithm:
    K_centered = K - row_means - col_means + grand_mean

Usage
-----
    from qie_research.analysis.cka_analysis import calculate_cka
    score = calculate_cka(X_quantum, X_classical)

Reference
---------
Kornblith, S., et al. (2019). "Similarity of Neural Network Representations 
Revisited." ICML.
"""

from __future__ import annotations

import numpy as np


def center_gram_matrix(K: np.ndarray) -> np.ndarray:
    """
    Apply double centering to a Gram matrix K.
    
    This is mathematically equivalent to computing H @ K @ H, where H is the 
    centering matrix (I - 1/n * 11^T). Centering ensures that the CKA 
    score is invariant to global shifts in the high-dimensional feature space.

    Implementation Note
    -------------------
    Direct matrix multiplication with H is O(n^3). This implementation uses
    broadcasted subtraction to achieve O(n^2) compute for centering.
    Note that end-to-end linear CKA here still materializes full n x n Gram
    matrices, so memory remains O(n^2) and is best suited to moderate sample
    sizes or sampled subsets of very large datasets.

    Parameters
    ----------
    K : np.ndarray of shape (n_samples, n_samples)
        The raw Gram matrix (inner product of features).

    Returns
    -------
    K_centered : np.ndarray of shape (n_samples, n_samples)
    """
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"Expected a square Gram matrix, got {K.shape}")

    n = K.shape[0]
    
    # Calculate means along rows, columns, and the entire matrix
    # (grand_mean must be scalar, row/col means must be vectors)
    row_means = np.mean(K, axis=0, keepdims=True)    # (1, n)
    col_means = np.mean(K, axis=1, keepdims=True)    # (n, 1)
    grand_mean = np.mean(K)                          # scalar

    # Apply centering formula: K_ij - mean(row_i) - mean(col_j) + mean(grand)
    return K - row_means - col_means + grand_mean


def compute_hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC).
    
    HSIC is a measure of the statistical dependence between two kernels. 
    In the context of CKA, it represents the 'raw' similarity before 
    normalization.

    Parameters
    ----------
    K, L : np.ndarray of shape (n_samples, n_samples)
        Centered or uncentered Gram matrices. Centering will be applied 
        automatically.

    Returns
    -------
    hsic_val : float
        The inner product of the centered kernels.
    """
    K_c = center_gram_matrix(K)
    L_c = center_gram_matrix(L)

    # The Frobenius inner product trace(K_c @ L_c) is equivalent to 
    # the sum of element-wise products. This is faster and avoids O(n^3).
    return float(np.sum(K_c * L_c))


def calculate_cka(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Calculate the Centered Kernel Alignment (CKA) between two feature maps.
    
    This is the primary public entry point for the analysis. It takes two 
    feature matrices (raw or encoded) and returns their geometric similarity.

    Parameters
    ----------
    X1 : np.ndarray of shape (n_samples, d1)
        The first feature matrix (e.g., Amplitude-encoded Wine data).
    X2 : np.ndarray of shape (n_samples, d2)
        The second feature matrix (e.g., RBF-baseline Wine data).
        n_samples MUST be identical for both X1 and X2.

    Returns
    -------
    cka_score : float
        A value between 0.0 and 1.0.
        - 1.0: The feature maps are geometrically identical (up to rotation/scaling).
        - 0.0: The feature maps represent completely different data relationships.

    Mathematical Interpretation
    ---------------------------
    A CKA score of ~0.95 suggests that your 'Quantum' encoding is just 
    replicating a classical baseline. A score of < 0.60 with high 
    classification accuracy suggests the Quantum encoding has found a 
    'Unique Geometric Advantage' for the task.
    """
    if X1.shape[0] != X2.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X1 has {X1.shape[0]}, X2 has {X2.shape[0]}. "
            "CKA requires identical data points in both sets."
        )

    # 1. Compute Linear Gram Matrices (The Kernels)
    # This represents the inner product of the features.
    print(f"  Calculating Gram matrices: K({X1.shape}) and L({X2.shape})...")
    K = X1 @ X1.T
    L = X2 @ X2.T

    # 2. Compute HSIC components
    # HSIC(K, L) is the covariance of the similarities.
    print("  Computing HSIC components...")
    hsic_kl = compute_hsic(K, L)
    hsic_kk = compute_hsic(K, K)
    hsic_ll = compute_hsic(L, L)

    # 3. Normalization
    # Similar to a Pearson correlation coefficient.
    denominator = np.sqrt(hsic_kk * hsic_ll)
    
    if not np.isfinite(denominator) or denominator <= 0 or np.isclose(denominator, 0.0):
        print(
            "  WARNING: Denominator is zero or numerically unstable "
            "(constant feature map or floating-point error). Returning 0.0."
        )
        return 0.0

    cka_score = hsic_kl / denominator
    if not np.isfinite(cka_score):
        print("  WARNING: CKA score is non-finite. Returning 0.0.")
        return 0.0
    print(f"  CKA score calculated: {cka_score:.6f}")
    
    return float(cka_score)


# -------------------------------------------------------------------------
# Demonstration / CLI Usage
# -------------------------------------------------------------------------

def main() -> None:
    """Run a quick sanity check for CKA calculation."""
    print("CKA Analysis Sanity Check")
    print("-" * 30)

    n_samples = 100
    d1 = 10
    d2 = 50

    # Test 1: Identity Case (Should be 1.0)
    X_id = np.random.randn(n_samples, d1)
    score_id = calculate_cka(X_id, X_id)
    print(f"Identity Test (X vs X): {score_id:.4f}")

    # Test 2: Rotation Invariance (Should be 1.0)
    # Generate an orthogonal matrix (Rotation)
    Q, _ = np.linalg.qr(np.random.randn(d1, d1))
    X_rot = X_id @ Q
    score_rot = calculate_cka(X_id, X_rot)
    print(f"Rotation Invariance Test (X vs rotated X): {score_rot:.4f}")

    # Test 3: Completely Different Data (Should be low)
    X_random = np.random.randn(n_samples, d2)
    score_rand = calculate_cka(X_id, X_random)
    print(f"Random Similarity Test (X vs noise): {score_rand:.4f}")


if __name__ == "__main__":
    main()
