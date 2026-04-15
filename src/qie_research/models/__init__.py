"""
Models
======
PyTorch training utilities for Phase 4 loss-curve and gradient-norm logging.

The sklearn-based models (logistic regression, RBF SVM, MLP) live in runner.py
because they are thin wrappers around sklearn objects with no training loop.
This package contains the differentiable training infrastructure that sklearn
cannot provide: per-epoch loss and gradient norm curves.
"""
