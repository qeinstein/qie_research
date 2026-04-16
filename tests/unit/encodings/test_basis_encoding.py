"""Unit tests for BasisEncoding."""

from __future__ import annotations

import numpy as np
import pytest

from qie_research.encodings.basis_encoding import BasisEncoding, _as_2d


# fixtures

@pytest.fixture()
def simple_X():
    rng = np.random.default_rng(1)
    return rng.uniform(0, 10, size=(8, 4))


@pytest.fixture()
def enc_fitted(simple_X):
    enc = BasisEncoding(n_bits=4)
    enc.fit(simple_X)
    return enc


# constructor validation

def test_invalid_n_bits_zero():
    with pytest.raises(ValueError, match="n_bits must be between"):
        BasisEncoding(n_bits=0)


def test_invalid_n_bits_too_large():
    with pytest.raises(ValueError, match="n_bits must be between"):
        BasisEncoding(n_bits=17)


def test_valid_n_bits_boundary_1():
    enc = BasisEncoding(n_bits=1)
    assert enc.n_bits == 1


def test_valid_n_bits_boundary_16():
    enc = BasisEncoding(n_bits=16)
    assert enc.n_bits == 16


# fit

def test_fit_sets_dimensions(simple_X):
    enc = BasisEncoding(n_bits=4)
    enc.fit(simple_X)
    assert enc.input_dim_ == 4
    assert enc.output_dim_ == 4 * 4


def test_fit_records_min_max(simple_X):
    enc = BasisEncoding(n_bits=4)
    enc.fit(simple_X)
    np.testing.assert_allclose(enc.min_, simple_X.min(axis=0))
    np.testing.assert_allclose(enc.max_, simple_X.max(axis=0))


def test_fit_constant_feature_no_div_zero():
    X = np.ones((5, 3))
    enc = BasisEncoding(n_bits=2)
    enc.fit(X)
    # _range_ must be 1.0 for constant features (not 0)
    assert np.all(enc._range_ == 1.0)


def test_fit_returns_self(simple_X):
    enc = BasisEncoding()
    assert enc.fit(simple_X) is enc


# transform

def test_transform_output_shape(enc_fitted, simple_X):
    X_enc = enc_fitted.transform(simple_X)
    assert X_enc.shape == (simple_X.shape[0], enc_fitted.output_dim_)


def test_transform_binary_values(enc_fitted, simple_X):
    X_enc = enc_fitted.transform(simple_X)
    assert np.all((X_enc == 0) | (X_enc == 1))


def test_transform_before_fit_raises():
    enc = BasisEncoding()
    with pytest.raises(RuntimeError, match="not fitted"):
        enc.transform(np.ones((2, 3)))


def test_transform_out_of_range_clips(enc_fitted, simple_X):
    # values beyond the training range should not produce non-binary outputs
    X_oor = simple_X.copy()
    X_oor[0, 0] = 9999.0
    X_oor[1, 0] = -9999.0
    X_enc = enc_fitted.transform(X_oor)
    assert np.all((X_enc == 0) | (X_enc == 1))


# verify_binary

def test_verify_binary_passes(enc_fitted, simple_X):
    X_enc = enc_fitted.transform(simple_X)
    assert enc_fitted.verify_binary(X_enc) is True


def test_verify_binary_fails_on_float():
    enc = BasisEncoding(n_bits=2)
    enc.fit(np.ones((2, 2)))
    assert enc.verify_binary(np.array([[0.5, 1.0]])) is False


# decode round-trip

def test_decode_roundtrip(simple_X):
    enc = BasisEncoding(n_bits=8)
    X_enc = enc.fit_transform(simple_X)
    X_int_recovered = enc.decode(X_enc)
    # round-trip: quantise manually and compare
    X_scaled = (simple_X - enc.min_) / enc._range_
    X_scaled = np.clip(X_scaled, 0.0, 1.0)
    X_int_expected = np.round(X_scaled * (enc.levels_ - 1)).astype(np.int32)
    np.testing.assert_array_equal(X_int_recovered, X_int_expected)


# fit_transform consistency

def test_fit_transform_matches_fit_then_transform(simple_X):
    enc1 = BasisEncoding(n_bits=4)
    X1 = enc1.fit_transform(simple_X)

    enc2 = BasisEncoding(n_bits=4)
    enc2.fit(simple_X)
    X2 = enc2.transform(simple_X)

    np.testing.assert_array_equal(X1, X2)


# __repr__

def test_repr():
    enc = BasisEncoding(n_bits=6)
    assert "BasisEncoding" in repr(enc)
    assert "n_bits=6" in repr(enc)


# parametrized output_dim

@pytest.mark.parametrize("d,n_bits,expected_out", [
    (3, 1, 3),
    (3, 8, 24),
    (5, 4, 20),
    (10, 2, 20),
])
def test_output_dim_parametrized(d, n_bits, expected_out):
    enc = BasisEncoding(n_bits=n_bits)
    enc.fit(np.zeros((2, d)))
    assert enc.output_dim_ == expected_out
