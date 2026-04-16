"""Unit tests for AmplitudeEncoding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qie_research.encodings.amplitude_encoding import AmplitudeEncoding, _as_2d


# fixtures

@pytest.fixture()
def simple_X():
    # 5 samples, 10 features — output_dim should be 16 (next power of two)
    rng = np.random.default_rng(0)
    return rng.standard_normal((5, 10))


@pytest.fixture()
def encoder_fitted(simple_X):
    enc = AmplitudeEncoding()
    enc.fit(simple_X)
    return enc


# _as_2d helper

def test_as_2d_1d_input():
    x = np.array([1.0, 2.0, 3.0])
    out = _as_2d(x)
    assert out.shape == (1, 3)


def test_as_2d_2d_passthrough():
    X = np.ones((4, 5))
    assert _as_2d(X).shape == (4, 5)


def test_as_2d_3d_raises():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        _as_2d(np.ones((2, 3, 4)))


# AmplitudeEncoding.fit

def test_fit_sets_input_dim(encoder_fitted, simple_X):
    assert encoder_fitted.input_dim_ == simple_X.shape[1]


def test_fit_sets_output_dim_power_of_two(encoder_fitted):
    # 10 features → ceil(log2(10)) = 4 → 2^4 = 16
    assert encoder_fitted.output_dim_ == 16


def test_fit_sets_n_qubits(encoder_fitted):
    assert encoder_fitted.n_qubits_ == 4


def test_fit_no_pad_output_dim_equals_input(simple_X):
    enc = AmplitudeEncoding(pad_to_power_of_two=False)
    enc.fit(simple_X)
    assert enc.output_dim_ == simple_X.shape[1]


def test_fit_returns_self(simple_X):
    enc = AmplitudeEncoding()
    result = enc.fit(simple_X)
    assert result is enc


def test_fit_accepts_1d(simple_X):
    enc = AmplitudeEncoding()
    enc.fit(simple_X[0])  # 1-D vector
    assert enc.input_dim_ == simple_X.shape[1]


def test_fit_single_feature():
    # edge: 1 feature → n_qubits=1, output_dim=2 (or 1 without padding)
    enc = AmplitudeEncoding(pad_to_power_of_two=True)
    enc.fit(np.array([[1.0]]))
    assert enc.n_qubits_ == 1
    assert enc.output_dim_ == 2


# AmplitudeEncoding.transform

def test_transform_output_shape(encoder_fitted, simple_X):
    X_enc = encoder_fitted.transform(simple_X)
    assert X_enc.shape == (simple_X.shape[0], encoder_fitted.output_dim_)


def test_transform_unit_norm(encoder_fitted, simple_X):
    X_enc = encoder_fitted.transform(simple_X)
    norms = np.linalg.norm(X_enc, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_transform_zero_norm_sample(encoder_fitted):
    # zero-norm sample must not raise; epsilon prevents div-by-zero
    X_zero = np.zeros((1, 10))
    X_enc = encoder_fitted.transform(X_zero)
    assert np.all(np.isfinite(X_enc))


def test_transform_before_fit_raises():
    enc = AmplitudeEncoding()
    with pytest.raises(RuntimeError, match="not fitted"):
        enc.transform(np.ones((3, 5)))


def test_transform_padding_zeros(encoder_fitted, simple_X):
    # columns beyond input_dim_ should be zero for non-padded portion
    X_enc = encoder_fitted.transform(simple_X)
    pad_start = encoder_fitted.input_dim_
    pad_end = encoder_fitted.output_dim_
    assert np.all(X_enc[:, pad_start:pad_end] == 0.0)


# fit_transform

def test_fit_transform_matches_fit_then_transform(simple_X):
    enc1 = AmplitudeEncoding()
    X1 = enc1.fit_transform(simple_X)

    enc2 = AmplitudeEncoding()
    enc2.fit(simple_X)
    X2 = enc2.transform(simple_X)

    np.testing.assert_array_equal(X1, X2)


# verify_normalization

def test_verify_normalization_passes(encoder_fitted, simple_X):
    X_enc = encoder_fitted.transform(simple_X)
    assert encoder_fitted.verify_normalization(X_enc) is True


def test_verify_normalization_fails_on_unnormalised(encoder_fitted, simple_X):
    X_enc = encoder_fitted.transform(simple_X) * 2.0
    assert encoder_fitted.verify_normalization(X_enc) is False


# __repr__

def test_repr_contains_class_name():
    enc = AmplitudeEncoding(pad_to_power_of_two=False, epsilon=1e-9)
    r = repr(enc)
    assert "AmplitudeEncoding" in r
    assert "pad_to_power_of_two=False" in r


# output_dim correctness for various input sizes

@pytest.mark.parametrize("d,expected_out", [
    (1, 2),   # ceil(log2(1))=0 → max(0,1)=1 → 2^1=2
    (2, 2),
    (3, 4),
    (4, 4),
    (5, 8),
    (16, 16),
    (17, 32),
])
def test_output_dim_parametrized(d, expected_out):
    enc = AmplitudeEncoding(pad_to_power_of_two=True)
    enc.fit(np.ones((2, d)))
    assert enc.output_dim_ == expected_out
