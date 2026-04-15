"""
Basis Encoding

Maps each continuous feature xᵢ to its b-bit binary representation,
mirroring the quantum computational basis encoding where a classical
bit string maps directly to a basis state:

    x = (x₁, x₂, ..., x_d)  →  |b₁ b₂ ... b_d⟩

where each bᵢ is the b-bit binary representation of the discretised xᵢ.

Classical feature map
---------------------
    1. Fit:       record per-feature min and max from training data
    2. Scale:     x̃ᵢ = (xᵢ - minᵢ) / (maxᵢ - minᵢ)  ∈ [0, 1]
    3. Quantise:  kᵢ = round(x̃ᵢ · (2^b - 1))          ∈ {0, ..., 2^b - 1}
    4. Encode:    bᵢ = binary(kᵢ)                       ∈ {0, 1}^b
    5. Output:    φ(x) = [b₁ ‖ b₂ ‖ ... ‖ b_d]        ∈ {0, 1}^{d · b}

Representational budget
-----------------------
Output dimension is d · b.  At the default b=8 this gives 256 resolution
levels per feature, matching an 8-bit classical discretisation.  One-hot
encoding (2^b bins per feature) is intentionally excluded: it would produce
d · 256 = O(10⁴) dimensions for typical tabular datasets, making matched-
budget comparisons against classical baselines impractical.

Differentiability
-----------------
This encoding is NOT differentiable.  The round() operation in the
quantisation step has zero gradient almost everywhere.  NTK analysis and
gradient-norm tracking do not apply to basis encoding.  This is a known
structural limitation and is reported as part of the benchmark findings.

Out-of-range handling
---------------------
Test samples outside [minᵢ, maxᵢ] are clipped before quantisation.
This prevents invalid bit patterns and is the only numerically safe option.
"""

from __future__ import annotations

import numpy as np


class BasisEncoding:
    """
    Basis encoding as an explicit classical feature map.

    Parameters
    ----------
    n_bits : int, default 8
        Number of bits used to represent each feature.  The output dimension
        is input_dim_ * n_bits.  Higher values give finer resolution at the
        cost of a larger feature vector.

    Attributes
    ----------
    input_dim_ : int
        Feature dimensionality seen during ``fit``.
    output_dim_ : int
        Always input_dim_ * n_bits.
    min_ : np.ndarray of shape (input_dim_,)
        Per-feature minimum from the training set.
    max_ : np.ndarray of shape (input_dim_,)
        Per-feature maximum from the training set.
    levels_ : int
        Number of discrete quantisation levels: 2^n_bits.
    """

    def __init__(self, n_bits: int = 8) -> None:
        if n_bits < 1 or n_bits > 16:
            raise ValueError(
                f"n_bits must be between 1 and 16, got {n_bits}."
            )
        self.n_bits = n_bits

        self.input_dim_: int | None = None
        self.output_dim_: int | None = None
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.levels_: int = 2 ** n_bits

    # Scikit-learn compatible interface

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BasisEncoding":
        """
        Record per-feature min and max from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, d) or (d,)
        y : ignored

        Returns
        -------
        self
        """
        X = _as_2d(X).astype(float)
        self.input_dim_ = X.shape[1]
        self.output_dim_ = self.input_dim_ * self.n_bits

        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)

        # Guard against constant features (max == min)
        self._range_ = np.where(
            self.max_ - self.min_ > 0,
            self.max_ - self.min_,
            1.0,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, d) or (d,)

        Returns
        -------
        X_enc : np.ndarray of shape (n_samples, d * n_bits), dtype float64
            Each block of n_bits columns is the binary representation of
            the corresponding discretised feature, MSB first.
        """
        self._check_is_fitted()
        X = _as_2d(X).astype(float)
        n_samples, d = X.shape

        # Scale to [0, 1], clip out-of-range test samples
        X_scaled = (X - self.min_) / self._range_
        X_scaled = np.clip(X_scaled, 0.0, 1.0)

        # Quantise to integer indices in {0, ..., levels_ - 1}
        X_int = np.round(X_scaled * (self.levels_ - 1)).astype(np.int32)

        # Unpack to binary bits: (n_samples, d, n_bits), MSB first
        bit_positions = np.arange(self.n_bits - 1, -1, -1, dtype=np.int32)
        bits = ((X_int[:, :, np.newaxis] >> bit_positions) & 1).astype(float)

        # Flatten to (n_samples, d * n_bits)
        X_enc = bits.reshape(n_samples, -1)

        return X_enc

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # Diagnostics (used in Phase 2 numerical audit)

    def verify_binary(self, X_enc: np.ndarray) -> bool:
        """
        Return True if every value in X_enc is exactly 0 or 1.

        Parameters
        ----------
        X_enc : np.ndarray of shape (n_samples, d * n_bits)
        """
        return bool(np.all((X_enc == 0) | (X_enc == 1)))

    def decode(self, X_enc: np.ndarray) -> np.ndarray:
        """
        Invert the binary encoding back to integer indices.

        Useful for verifying round-trip fidelity during the Phase 2 audit.

        Parameters
        ----------
        X_enc : np.ndarray of shape (n_samples, d * n_bits)

        Returns
        -------
        X_int : np.ndarray of shape (n_samples, d)
            Recovered integer quantisation indices.
        """
        self._check_is_fitted()
        n_samples = X_enc.shape[0]
        bits = X_enc.reshape(n_samples, self.input_dim_, self.n_bits).astype(np.int32)
        bit_positions = np.arange(self.n_bits - 1, -1, -1, dtype=np.int32)
        X_int = np.sum(bits * bit_positions[::-1].reshape(1, 1, -1), axis=2)

        # Reconstruct using: bit_positions are already MSB-first powers of 2
        powers = (2 ** np.arange(self.n_bits - 1, -1, -1)).reshape(1, 1, -1)
        X_int = np.sum(bits * powers, axis=2)
        return X_int

    # Internal helpers

    def _check_is_fitted(self) -> None:
        if self.input_dim_ is None:
            raise RuntimeError(
                "BasisEncoding is not fitted. Call fit() before transform()."
            )

    def __repr__(self) -> str:
        return f"BasisEncoding(n_bits={self.n_bits})"


# Module-level helper

def _as_2d(X: np.ndarray) -> np.ndarray:
    """Ensure X is a 2-D array, reshaping a 1-D vector to (1, d)."""
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(
            f"Expected a 1-D or 2-D array, got shape {X.shape}."
        )
    return X
