"""
Amplitude Encoding

Maps a classical vector x ∈ R^d into a unit-norm state vector padded to the
next power of two, mirroring the quantum amplitude representation:

    |ψ⟩ = ∑ᵢ cᵢ |i⟩,   subject to  ∑ᵢ |cᵢ|² = 1

Classical feature map
---------------------
    1. Normalise:  x̂  = x / (‖x‖₂ + ε)
    2. Pad:        φ(x) ∈ R^{2ⁿ},  n = ⌈log₂(d)⌉,  zeros appended
    3. Verify:     ∑ᵢ φ(x)ᵢ² = 1

Representational budget
-----------------------
For d input features the output dimension is 2^⌈log₂(d)⌉.  This expansion
(e.g. d=10 → 16, d=100 → 128) is the claim under test in this benchmark.
All classical baselines must be matched to this same d_out.

This encoding is differentiable, so NTK analysis applies directly.
"""

from __future__ import annotations

import math

import numpy as np


class AmplitudeEncoding:
    """
    Amplitude encoding as an explicit, differentiable classical feature map.

    Parameters
    ----------
    pad_to_power_of_two : bool, default True
        Pad the normalised vector to the next power of two, matching the
        qubit-register interpretation.  Set to False only for ablations where
        you want normalisation without dimensional expansion.
    epsilon : float, default 1e-12
        Added to the L2 norm before dividing to avoid division by zero for
        degenerate zero-norm samples.

    Attributes
    ----------
    input_dim_ : int
        Feature dimensionality seen during ``fit``.
    output_dim_ : int
        Dimensionality of the encoded output (2^⌈log₂(input_dim_)⌉ or
        input_dim_ when pad_to_power_of_two=False).
    n_qubits_ : int
        Number of qubits implied by the encoding (⌈log₂(input_dim_)⌉).
    """

    def __init__(
        self,
        pad_to_power_of_two: bool = True,
        epsilon: float = 1e-12,
    ) -> None:
        self.pad_to_power_of_two = pad_to_power_of_two
        self.epsilon = epsilon  # prevents "division by zero" errors

        # "_" indicates that these values are not set during initialization
        self.input_dim_: int | None = None
        self.output_dim_: int | None = None
        self.n_qubits_: int | None = None

    # Scikit-learn compatible interface

    # the fit method implements the logic for preparing an Amplitude Encoding scheme
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AmplitudeEncoding":
        """
        Record input dimensionality and compute output dimensionality.

        Parameters
        ----------
        X : array-like of shape (n_samples, d) or (d,)
        y : ignored

        Returns
        -------
        self
        """
        X = _as_2d(X)
        self.input_dim_ = X.shape[1]

        n_qubits = max(1, math.ceil(math.log2(self.input_dim_)))
        self.n_qubits_ = n_qubits
        self.output_dim_ = (
            2 ** n_qubits if self.pad_to_power_of_two else self.input_dim_
        )
        return self


    # conversion of classical data into a format suitable for Amplitude Encoding.
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, d) or (d,)

        Returns
        -------
        X_enc : np.ndarray of shape (n_samples, output_dim_)
            Each row is L2-normalised and zero-padded to output_dim_.
        """
        self._check_is_fitted()
        X = _as_2d(X).astype(float)

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + self.epsilon)

        if self.pad_to_power_of_two and self.output_dim_ > self.input_dim_:
            pad = self.output_dim_ - self.input_dim_
            X_norm = np.pad(X_norm, ((0, 0), (0, pad)), mode="constant")

        return X_norm

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # Diagnostics (used in Phase 2 numerical audit)

    def verify_normalization(
        self, X_enc: np.ndarray, atol: float = 1e-6
    ) -> bool:
        """
        Return True if every row of X_enc satisfies ∑ᵢ xᵢ² ≈ 1.

        Parameters
        ----------
        X_enc : np.ndarray of shape (n_samples, output_dim_)
        atol : float
            Absolute tolerance passed to np.allclose.
        """
        squared_norms = np.sum(X_enc ** 2, axis=1)
        return bool(np.allclose(squared_norms, 1.0, atol=atol))

    # Internal helpers
    def _check_is_fitted(self) -> None:
        if self.input_dim_ is None:
            raise RuntimeError(
                "AmplitudeEncoding is not fitted. Call fit() before transform()."
            )

    def __repr__(self) -> str:
        return (
            f"AmplitudeEncoding("
            f"pad_to_power_of_two={self.pad_to_power_of_two}, "
            f"epsilon={self.epsilon})"
        )


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
