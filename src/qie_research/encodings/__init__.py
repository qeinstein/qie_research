"""
Quantum-inspired feature encodings.

All three encodings share a consistent scikit-learn compatible interface:

    encoder.fit(X_train)
    X_enc = encoder.transform(X)

The registry maps config-level string names to encoder classes, allowing
the config-driven runner to instantiate any encoding by name.

Encoding summary
----------------
Encoding     Output dim          Differentiable
-----------  ------------------  --------------
amplitude    2^⌈log₂(d)⌉         Yes
angle        2d                  Yes
basis        d * n_bits          No
"""

from qie_research.encodings.amplitude_encoding import AmplitudeEncoding
from qie_research.encodings.angle_encoding import AngleEncoding
from qie_research.encodings.basis_encoding import BasisEncoding

__all__ = [
    "AmplitudeEncoding",
    "AngleEncoding",
    "BasisEncoding",
    "ENCODING_REGISTRY",
]

ENCODING_REGISTRY: dict[str, type] = {
    "amplitude": AmplitudeEncoding,
    "angle": AngleEncoding,
    "basis": BasisEncoding,
}
