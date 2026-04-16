"""Unit tests for the encoding registry and shared _as_2d helpers."""

from __future__ import annotations

import numpy as np
import pytest

from qie_research.encodings import ENCODING_REGISTRY, AmplitudeEncoding, AngleEncoding, BasisEncoding


def test_registry_amplitude_class():
    assert ENCODING_REGISTRY["amplitude"] is AmplitudeEncoding


def test_registry_angle_class():
    assert ENCODING_REGISTRY["angle"] is AngleEncoding


def test_registry_basis_class():
    assert ENCODING_REGISTRY["basis"] is BasisEncoding


def test_registry_instantiate_amplitude():
    enc = ENCODING_REGISTRY["amplitude"](pad_to_power_of_two=True)
    assert isinstance(enc, AmplitudeEncoding)


def test_registry_instantiate_angle():
    enc = ENCODING_REGISTRY["angle"](standardize=False)
    assert isinstance(enc, AngleEncoding)


def test_registry_instantiate_basis():
    enc = ENCODING_REGISTRY["basis"](n_bits=4)
    assert isinstance(enc, BasisEncoding)


@pytest.mark.parametrize("enc_name", ["amplitude", "angle", "basis"])
def test_all_encodings_sklearn_interface(enc_name):
    # every encoding must support fit / transform / fit_transform
    rng = np.random.default_rng(99)
    X = rng.uniform(0, 5, (12, 6))
    enc = ENCODING_REGISTRY[enc_name]()
    enc.fit(X)
    X_enc = enc.transform(X)
    assert X_enc.shape[0] == X.shape[0]
    assert hasattr(enc, "input_dim_")
    assert hasattr(enc, "output_dim_")
    assert enc.output_dim_ == X_enc.shape[1]


@pytest.mark.parametrize("enc_name", ["amplitude", "angle", "basis"])
def test_all_encodings_fit_transform_consistent(enc_name):
    rng = np.random.default_rng(100)
    X = rng.standard_normal((8, 5))
    enc = ENCODING_REGISTRY[enc_name]()
    X1 = enc.fit_transform(X)

    enc2 = ENCODING_REGISTRY[enc_name]()
    enc2.fit(X)
    X2 = enc2.transform(X)

    np.testing.assert_array_equal(X1, X2)
