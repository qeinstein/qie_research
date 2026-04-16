"""
Angle Encoding

Maps each scalar feature xᵢ to a pair of trigonometric values derived from
the quantum rotation gate Ry(θᵢ):

    Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩

Reading off the amplitudes gives the per-feature classical map:

    φ(xᵢ) = [cos(θᵢ / 2),  sin(θᵢ / 2)]

where  θᵢ = scale · xᵢ,  with scale = π by default.  To achieve full
angular coverage (θᵢ ∈ [-π, π]) without aliasing, features must lie in
[-1, 1] before multiplication.  When standardize=True, a min-max transform
is applied per feature so that the training range maps exactly to [-1, 1].
This guarantees θᵢ ∈ [-π, π] for in-distribution samples.  Z-score
standardization is intentionally NOT used: it does not bound the output,
so outliers silently alias to a different angle with no error.

The full map doubles the dimensionality:

    φ(x) ∈ R^{2d},  interleaved as  [cos(θ₁/2), sin(θ₁/2), ..., cos(θ_d/2), sin(θ_d/2)]

Representational budget
-----------------------
Output dimension is exactly 2d.  Match classical baselines to this d_out.

Differentiability
-----------------
The map is smooth and differentiable everywhere, so NTK analysis applies.

Implementation note
-------------------
Dense angle encoding (two features per qubit via Ry + Rz) is intentionally
excluded.  That variant introduces feature cross-products, which conflates
the encoding geometry with interaction effects and obscures the clean
geometric analysis this benchmark requires.
"""

from __future__ import annotations

import math

import numpy as np


class AngleEncoding:
    """
    Angle encoding as an explicit, differentiable classical feature map.

    Parameters
    ----------
    scale : float, default math.pi
        Multiplier applied to each feature before the trigonometric map:
        θᵢ = scale · xᵢ.  The default π, combined with standardize=True
        (which maps each feature to [-1, 1]), gives θᵢ ∈ [-π, π] and
        full angular coverage with no aliasing.
    standardize : bool, default True
        If True, each feature is min-max scaled to [-1, 1] during fit,
        and the same statistics are applied during transform.  Set to
        False only when the caller guarantees the input is already in
        [-1, 1].

    Attributes
    ----------
    input_dim_ : int
        Feature dimensionality seen during ``fit``.
    output_dim_ : int
        Always 2 * input_dim_.
    min_ : np.ndarray of shape (input_dim_,)
        Per-feature minimum from the training set (only set when
        standardize=True).
    range_ : np.ndarray of shape (input_dim_,)
        Per-feature range (max - min) from the training set, floored at
        1e-8 for constant features (only set when standardize=True).
    """

    def __init__(
        self,
        scale: float = math.pi,
        standardize: bool = True,
    ) -> None:
        self.scale = scale
        self.standardize = standardize

        self.input_dim_: int | None = None
        self.output_dim_: int | None = None
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    # Scikit-learn compatible interface

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AngleEncoding":
        """
        Record input dimensionality and, if standardize=True, compute
        per-feature min and range from X.

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
        self.output_dim_ = 2 * self.input_dim_

        if self.standardize:
            self.min_ = X.min(axis=0)
            # Floor prevents division by zero for constant features.
            self.range_ = np.maximum(X.max(axis=0) - X.min(axis=0), 1e-8)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, d) or (d,)

        Returns
        -------
        X_enc : np.ndarray of shape (n_samples, 2 * d)
            Rows are interleaved cos/sin pairs:
            [cos(θ₁/2), sin(θ₁/2), cos(θ₂/2), sin(θ₂/2), ...]
        """
        self._check_is_fitted()
        X = _as_2d(X).astype(float)

        if self.standardize:
            # Min-max to [-1, 1]: maps training range to exactly [-1, 1].
            # Out-of-distribution test values may exceed [-1, 1] and alias;
            # this is unavoidable without clipping (which would lose rank).
            X = 2.0 * (X - self.min_) / self.range_ - 1.0

        theta = self.scale * X                     # (n_samples, d)
        half_theta = theta / 2.0                   # Ry gate uses θ/2

        cos_vals = np.cos(half_theta)              # (n_samples, d)
        sin_vals = np.sin(half_theta)              # (n_samples, d)

        # Interleave: [cos(θ₁/2), sin(θ₁/2), cos(θ₂/2), sin(θ₂/2), ...]
        n_samples, d = cos_vals.shape
        X_enc = np.empty((n_samples, 2 * d), dtype=float)
        X_enc[:, 0::2] = cos_vals
        X_enc[:, 1::2] = sin_vals

        return X_enc

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # Diagnostics (used in Phase 2 numerical audit)

    def verify_unit_circle(
        self, X_enc: np.ndarray, atol: float = 1e-6
    ) -> bool:
        """
        Return True if every cos/sin pair lies on the unit circle, i.e.
        cos²(θᵢ/2) + sin²(θᵢ/2) = 1 for all i.

        Parameters
        ----------
        X_enc : np.ndarray of shape (n_samples, 2 * d)
        atol : float
        """
        cos_vals = X_enc[:, 0::2]
        sin_vals = X_enc[:, 1::2]
        pair_norms = cos_vals ** 2 + sin_vals ** 2
        return bool(np.allclose(pair_norms, 1.0, atol=atol))

    # Internal helpers

    def _check_is_fitted(self) -> None:
        if self.input_dim_ is None:
            raise RuntimeError(
                "AngleEncoding is not fitted. Call fit() before transform()."
            )

    def __repr__(self) -> str:
        return (
            f"AngleEncoding("
            f"scale={self.scale}, "
            f"standardize={self.standardize})"
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
