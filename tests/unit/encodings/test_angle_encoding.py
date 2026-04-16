"""Unit tests for AngleEncoding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qie_research.encodings.angle_encoding import AngleEncoding, _as_2d


# fixtures

@pytest.fixture()
def simple_X():
    rng = np.random.default_rng(2)
    return rng.standard_normal((6, 5))


@pytest.fixture()
def enc_fitted(simple_X):
    enc = AngleEncoding()
    enc.fit(simple_X)
    return enc


# fit

def test_fit_sets_input_dim(enc_fitted, simple_X):
    assert enc_fitted.input_dim_ == simple_X.shape[1]


def test_fit_sets_output_dim(enc_fitted, simple_X):
    assert enc_fitted.output_dim_ == 2 * simple_X.shape[1]


def test_fit_computes_min_range_when_standardize(simple_X):
    enc = AngleEncoding(standardize=True)
    enc.fit(simple_X)
    np.testing.assert_allclose(enc.min_, simple_X.min(axis=0))
    expected_range = np.maximum(simple_X.max(axis=0) - simple_X.min(axis=0), 1e-8)
    np.testing.assert_allclose(enc.range_, expected_range)


def test_fit_skips_min_range_when_no_standardize(simple_X):
    enc = AngleEncoding(standardize=False)
    enc.fit(simple_X)
    assert enc.min_ is None
    assert enc.range_ is None


def test_fit_constant_feature_range_floor():
    X = np.ones((5, 3))
    enc = AngleEncoding(standardize=True)
    enc.fit(X)
    # Constant features have range=0; floored to 1e-8 to prevent division by zero.
    assert np.all(enc.range_ >= 1e-8)


def test_fit_returns_self(simple_X):
    enc = AngleEncoding()
    assert enc.fit(simple_X) is enc


# transform

def test_transform_output_shape(enc_fitted, simple_X):
    X_enc = enc_fitted.transform(simple_X)
    assert X_enc.shape == (simple_X.shape[0], enc_fitted.output_dim_)


def test_transform_unit_circle(enc_fitted, simple_X):
    X_enc = enc_fitted.transform(simple_X)
    cos_vals = X_enc[:, 0::2]
    sin_vals = X_enc[:, 1::2]
    pair_norms = cos_vals ** 2 + sin_vals ** 2
    np.testing.assert_allclose(pair_norms, 1.0, atol=1e-12)


def test_transform_before_fit_raises():
    enc = AngleEncoding()
    with pytest.raises(RuntimeError, match="not fitted"):
        enc.transform(np.ones((3, 4)))


def test_transform_interleave_order(simple_X):
    # even cols = cos, odd cols = sin
    enc = AngleEncoding(standardize=False, scale=math.pi)
    enc.fit(simple_X)
    X_enc = enc.transform(simple_X)
    theta_half = (math.pi * simple_X) / 2.0
    np.testing.assert_allclose(X_enc[:, 0::2], np.cos(theta_half), atol=1e-12)
    np.testing.assert_allclose(X_enc[:, 1::2], np.sin(theta_half), atol=1e-12)


def test_transform_scale_effect():
    # with scale=0 all angles are 0, so cos=1 and sin=0
    X = np.ones((3, 2))
    enc = AngleEncoding(scale=0.0, standardize=False)
    enc.fit(X)
    X_enc = enc.transform(X)
    np.testing.assert_allclose(X_enc[:, 0::2], 1.0, atol=1e-12)
    np.testing.assert_allclose(X_enc[:, 1::2], 0.0, atol=1e-12)


# fit_transform consistency

def test_fit_transform_matches_fit_then_transform(simple_X):
    enc1 = AngleEncoding()
    X1 = enc1.fit_transform(simple_X)

    enc2 = AngleEncoding()
    enc2.fit(simple_X)
    X2 = enc2.transform(simple_X)

    np.testing.assert_array_equal(X1, X2)


# verify_unit_circle

def test_verify_unit_circle_passes(enc_fitted, simple_X):
    X_enc = enc_fitted.transform(simple_X)
    assert enc_fitted.verify_unit_circle(X_enc) is True


def test_verify_unit_circle_fails_on_scaled(enc_fitted, simple_X):
    X_enc = enc_fitted.transform(simple_X) * 2.0
    assert enc_fitted.verify_unit_circle(X_enc) is False


# __repr__

def test_repr():
    enc = AngleEncoding(scale=1.5, standardize=False)
    r = repr(enc)
    assert "AngleEncoding" in r
    assert "scale=1.5" in r
    assert "standardize=False" in r


# 1-D input accepted

def test_fit_transform_1d_input():
    enc = AngleEncoding()
    x = np.array([1.0, 2.0, 3.0])
    X_enc = enc.fit_transform(x)
    assert X_enc.shape == (1, 6)
