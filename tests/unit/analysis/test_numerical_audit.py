"""Unit tests for analysis/numerical_audit.py helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from qie_research.analysis.numerical_audit import (
    _condition_number,
    _effective_rank,
    _noise_stability,
    _singular_values,
    _verify_invariants,
)
from qie_research.encodings.amplitude_encoding import AmplitudeEncoding
from qie_research.encodings.angle_encoding import AngleEncoding
from qie_research.encodings.basis_encoding import BasisEncoding


# _singular_values

def test_singular_values_descending():
    rng = np.random.default_rng(10)
    X = rng.standard_normal((20, 5))
    sigma = _singular_values(X)
    assert np.all(np.diff(sigma) <= 0), "singular values should be non-increasing"


def test_singular_values_count():
    X = np.eye(4)
    sigma = _singular_values(X)
    assert len(sigma) == 4


# _effective_rank

def test_effective_rank_identity_is_full():
    # identity matrix has uniform singular values → max erank = n
    X = np.eye(5)
    sigma = _singular_values(X)
    erank = _effective_rank(sigma)
    assert abs(erank - 5.0) < 0.5


def test_effective_rank_rank1_is_one():
    # rank-1 matrix has one non-zero singular value → erank ≈ 1
    X = np.outer(np.ones(5), np.ones(5))
    sigma = _singular_values(X)
    erank = _effective_rank(sigma)
    assert abs(erank - 1.0) < 0.1


def test_effective_rank_positive():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((15, 6))
    sigma = _singular_values(X)
    assert _effective_rank(sigma) > 0


# _condition_number

def test_condition_number_identity_is_one():
    sigma = np.ones(4)
    assert _condition_number(sigma) == pytest.approx(1.0)


def test_condition_number_single_nonzero():
    sigma = np.array([5.0, 0.0])
    assert _condition_number(sigma) == float("inf")


def test_condition_number_positive():
    sigma = np.array([10.0, 5.0, 2.0, 0.5])
    kappa = _condition_number(sigma)
    assert kappa == pytest.approx(20.0)


# _noise_stability

def test_noise_stability_zero_noise():
    enc = AmplitudeEncoding()
    X = np.random.default_rng(12).standard_normal((10, 6))
    enc.fit(X)
    rng = np.random.default_rng(12)
    # zero noise_std → stability should be ~0
    delta = _noise_stability(enc, X, noise_std=0.0, rng=rng)
    assert delta == pytest.approx(0.0, abs=1e-10)


def test_noise_stability_nonnegative():
    enc = AngleEncoding()
    X = np.random.default_rng(13).standard_normal((10, 4))
    enc.fit(X)
    rng = np.random.default_rng(13)
    delta = _noise_stability(enc, X, noise_std=0.01, rng=rng)
    assert delta >= 0.0


# _verify_invariants

def test_verify_amplitude_pass():
    enc = AmplitudeEncoding()
    X = np.random.default_rng(14).standard_normal((5, 4))
    X_enc = enc.fit_transform(X)
    result = _verify_invariants("amplitude", enc, X_enc)
    assert result["passed"] is True


def test_verify_angle_pass():
    enc = AngleEncoding()
    X = np.random.default_rng(15).standard_normal((5, 4))
    X_enc = enc.fit_transform(X)
    result = _verify_invariants("angle", enc, X_enc)
    assert result["passed"] is True


def test_verify_basis_pass():
    enc = BasisEncoding(n_bits=4)
    X = np.random.default_rng(16).uniform(0, 1, (5, 3))
    X_enc = enc.fit_transform(X)
    result = _verify_invariants("basis", enc, X_enc)
    assert result["passed"] is True


def test_verify_unknown_encoding():
    enc = AmplitudeEncoding()
    X_enc = np.ones((3, 4))
    result = _verify_invariants("unknown_encoding", enc, X_enc)
    assert result["passed"] is False
