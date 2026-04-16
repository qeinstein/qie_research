"""
PyTorch Training Utilities
==========================
Shared training loop used in two modes during Phase 4:

  Mode A — QIE encoding + linear head
    Encodes features using a frozen QIE encoder (amplitude, angle, or basis),
    then trains a single nn.Linear layer on the encoded output.  Gradients
    flow through the linear head only — the encoding step is a fixed classical
    transform and is never differentiated through.  For basis encoding this is
    a structural property: the quantisation step is discontinuous, so no
    gradient definition exists for it.  This is documented in each run's
    output JSON under ``encoding_note``.

  Mode B — end-to-end MLP
    Applies StandardScaler to raw features, then trains a fully-connected MLP
    end-to-end.  Gradients flow through all layers.  This is the differentiable
    counterpart to the sklearn MLPClassifier baseline and provides the loss /
    gradient-norm reference curves for the MLP comparator.

Both modes produce the same output structure so that Phase 5 analysis can
compare curves across methods without special-casing.

GPU Support
-----------
Both functions move tensors and models to CUDA automatically if a GPU is
available.  No code changes are needed between local CPU runs and Colab GPU
runs.

Output Schema
-------------
Each function returns a dict with the following keys:

    metrics : dict
        accuracy and f1_macro on the test split.

    training_curves : dict
        epochs       : list[int]  — epoch indices 1..N
        train_loss   : list[float] — mean cross-entropy loss per epoch
        grad_norm    : list[float] — L2 gradient norm of the last mini-batch
                       per epoch.  Reported per-batch (not averaged across
                       batches) because the mean of per-batch norms is not a
                       valid quantity for convergence analysis.

    gradients_through_encoding : bool
        False for Mode A (linear head only).
        True for Mode B (end-to-end MLP).

    encoding_note : str or None
        Human-readable note about gradient flow.  Always present for Mode A;
        None for Mode B.

    timing_seconds : float
        Wall-clock time for the full training run.

    memory_bytes : int
        Peak memory allocated during training.  On CUDA, measured via
        torch.cuda.max_memory_allocated() (device memory only; host-side
        allocation not tracked).  On CPU, measured via tracemalloc (Python
        heap only; PyTorch tensor storage allocated via the C++ allocator is
        not captured, so this number is a lower bound on true RSS growth).
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
except ImportError:
    raise ImportError(
        "PyTorch is required for torch_trainer. "
        "Install with: pip install torch\n"
        "In Colab: !pip install torch"
    )

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _grad_norm(model: nn.Module) -> float:
    """
    L2 norm of all gradients across all parameters.

    Equivalent to the value used by torch.nn.utils.clip_grad_norm_ when
    computing the total gradient norm before clipping.  Computed after
    loss.backward() and before optimizer.step().

    Returns 0.0 if no parameter has a gradient (should not occur during
    normal training).
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().norm(2).item() ** 2
    return float(total_sq ** 0.5)

class _MemoryTracker:
    """
    Context manager that measures peak memory allocated during a training block.

    On CUDA: uses torch.cuda.max_memory_allocated() (device memory).
    On CPU: uses tracemalloc (Python heap only; PyTorch C++ tensor storage
    is not captured, so the reported value is a lower bound).
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._peak: int = 0

    def __enter__(self) -> "_MemoryTracker":
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        else:
            tracemalloc.start()
        return self

    def __exit__(self, *_) -> None:
        if self._device.type == "cuda":
            self._peak = torch.cuda.max_memory_allocated(self._device)
        else:
            _, self._peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

    @property
    def peak_bytes(self) -> int:
        return self._peak

def train_linear_head(
    X_train_enc: np.ndarray,
    y_train: np.ndarray,
    X_test_enc: np.ndarray,
    y_test: np.ndarray,
    *,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 42,
    encoding_name: str = "unknown",
) -> dict:
    """
    Train a single nn.Linear layer on pre-encoded (frozen) features.

    Parameters
    ----------
    X_train_enc : np.ndarray, shape (n_train, d_enc)
        Encoded training features.  The encoding is already applied;
        this function does not touch the encoder.
    y_train : np.ndarray, shape (n_train,)
        Integer class labels.
    X_test_enc : np.ndarray, shape (n_test, d_enc)
        Encoded test features.
    y_test : np.ndarray, shape (n_test,)
        Integer class labels.
    n_epochs : int
        Number of full passes through the training data.
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 regularisation coefficient (Adam ``weight_decay``).
    batch_size : int
        Mini-batch size.
    seed : int
        Seed for torch random number generator.
    encoding_name : str
        Name of the QIE encoding used to produce X_train_enc.  Used to
        generate the ``encoding_note`` string in the output.

    Returns
    -------
    dict — see module docstring for schema.
    """
    torch.manual_seed(seed)
    device = _device()

    n_classes = int(len(np.unique(y_train)))
    d_in = int(X_train_enc.shape[1])

    # Cast to float32; basis encoding output is uint8.
    X_tr = torch.from_numpy(X_train_enc.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.int64)).to(device)
    X_te = torch.from_numpy(X_test_enc.astype(np.float32)).to(device)
    y_te_np = y_test.astype(np.int64)

    model = nn.Linear(d_in, n_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    grad_norms: list[float] = []

    t0 = time.perf_counter()
    with _MemoryTracker(device) as mem:
        model.train()
        for _ in range(n_epochs):
            perm = torch.randperm(len(X_tr), device=device)
            epoch_loss = 0.0
            last_gn = 0.0
            n_batches = 0

            for i in range(0, len(X_tr), batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = X_tr[idx], y_tr[idx]

                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                last_gn = _grad_norm(model)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_losses.append(round(epoch_loss / n_batches, 6))
            # Report the last batch's gradient norm per epoch.
            # Averaging norms across batches is not a meaningful quantity
            # for convergence analysis (mean(‖∇Lᵢ‖) ≠ ‖mean(∇Lᵢ)‖).
            grad_norms.append(round(last_gn, 6))

    elapsed = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).argmax(dim=1).cpu().numpy()

    if encoding_name == "basis":
        note = (
            "Gradients flow through the linear head only. "
            "Basis encoding uses discontinuous quantisation — no gradient "
            "definition exists for the encoding step by construction."
        )
    else:
        note = (
            "Gradients flow through the linear head only. "
            f"The {encoding_name} encoding is a fixed classical transform "
            "applied before training; it is not differentiated through."
        )

    return {
        "metrics": {
            "accuracy": round(float(accuracy_score(y_te_np, y_pred)), 6),
            "f1_macro": round(
                float(f1_score(y_te_np, y_pred, average="macro", zero_division=0)),
                6,
            ),
        },
        "training_curves": {
            "epochs": list(range(1, n_epochs + 1)),
            "train_loss": train_losses,
            "grad_norm": grad_norms,
        },
        "gradients_through_encoding": False,
        "encoding_note": note,
        "timing_seconds": round(elapsed, 6),
        "memory_bytes": mem.peak_bytes,
    }

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_layer_sizes: list[int] | tuple[int, ...] = (256, 128),
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> dict:
    """
    Train a fully-connected MLP end-to-end on StandardScaler-normalised features.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, d)
        Raw (unscaled) training features.  StandardScaler is applied internally.
    y_train : np.ndarray, shape (n_train,)
        Integer class labels.
    X_test : np.ndarray, shape (n_test, d)
        Raw test features.  The scaler fitted on X_train is applied.
    y_test : np.ndarray, shape (n_test,)
        Integer class labels.
    hidden_layer_sizes : sequence of int
        Sizes of hidden layers.  Default [256, 128] matches the sklearn MLP
        baseline architecture so that comparisons are architecturally fair.
    n_epochs : int
        Number of full passes through the training data.
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 regularisation coefficient.
    batch_size : int
        Mini-batch size.
    seed : int
        Seed for torch random number generator.
    max_samples : int, optional
        If set and n_train > max_samples, subsample training data
        (stratified by seed).  Used to keep large-dataset runs tractable.

    Returns
    -------
    dict — see module docstring for schema.
    """
    torch.manual_seed(seed)
    device = _device()

    # Optional subsampling
    if max_samples is not None and len(X_train) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), size=int(max_samples), replace=False)
        idx.sort()
        X_train, y_train = X_train[idx], y_train[idx]
        subsampled_n = int(max_samples)
    else:
        subsampled_n = None

    # Scale features
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_te_scaled = scaler.transform(X_test).astype(np.float32)

    n_classes = int(len(np.unique(y_train)))
    d_in = int(X_tr_scaled.shape[1])

    X_tr = torch.from_numpy(X_tr_scaled).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.int64)).to(device)
    X_te = torch.from_numpy(X_te_scaled).to(device)
    y_te_np = y_test.astype(np.int64)

    # Build MLP: Linear → ReLU → ... → Linear (output)
    layers: list[nn.Module] = []
    in_size = d_in
    for h in hidden_layer_sizes:
        layers.append(nn.Linear(in_size, h))
        layers.append(nn.ReLU())
        in_size = h
    layers.append(nn.Linear(in_size, n_classes))
    model = nn.Sequential(*layers).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    grad_norms: list[float] = []

    t0 = time.perf_counter()
    with _MemoryTracker(device) as mem:
        model.train()
        for _ in range(n_epochs):
            perm = torch.randperm(len(X_tr), device=device)
            epoch_loss = 0.0
            last_gn = 0.0
            n_batches = 0

            for i in range(0, len(X_tr), batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = X_tr[idx], y_tr[idx]

                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                last_gn = _grad_norm(model)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_losses.append(round(epoch_loss / n_batches, 6))
            grad_norms.append(round(last_gn, 6))

    elapsed = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).argmax(dim=1).cpu().numpy()

    result = {
        "metrics": {
            "accuracy": round(float(accuracy_score(y_te_np, y_pred)), 6),
            "f1_macro": round(
                float(f1_score(y_te_np, y_pred, average="macro", zero_division=0)),
                6,
            ),
        },
        "training_curves": {
            "epochs": list(range(1, n_epochs + 1)),
            "train_loss": train_losses,
            "grad_norm": grad_norms,
        },
        "gradients_through_encoding": True,
        "encoding_note": None,
        "timing_seconds": round(elapsed, 6),
        "memory_bytes": mem.peak_bytes,
    }
    if subsampled_n is not None:
        result["subsampled_train_n"] = subsampled_n
    return result
