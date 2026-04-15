# [PHASE 1] Infrastructure and Determinism

- Record type: phase-outcome
- Phase: 1 - Infrastructure and determinism
- Status: Ready for review
- Owner: Fluxx
- Related branch or Pull Request: `phase-1-implementation`

## Objective

Build a config-driven runner, enforce strict seed control, add logging for
metrics and overhead, and verify that a full comparison path is reproducible
end to end by a teammate from a single command.

## Deliverables

### 1. Encoding implementations

Three quantum-inspired encodings implemented as explicit classical feature
maps in [`src/qie_research/encodings/`](../../src/qie_research/encodings/):

| File | Class | Output dim | Differentiable |
|---|---|---|---|
| [`amplitude_encoding.py`](../../src/qie_research/encodings/amplitude_encoding.py) | `AmplitudeEncoding` | 2^⌈log₂(d)⌉ | Yes |
| [`angle_encoding.py`](../../src/qie_research/encodings/angle_encoding.py) | `AngleEncoding` | 2d | Yes |
| [`basis_encoding.py`](../../src/qie_research/encodings/basis_encoding.py) | `BasisEncoding` | d · n_bits | No |

All three expose a consistent scikit-learn compatible interface:
`fit`, `transform`, `fit_transform`.  Each includes a diagnostic method
for use in the Phase 2 numerical audit: `verify_normalization`,
`verify_unit_circle`, and `verify_binary` respectively.

A shared `ENCODING_REGISTRY` in [`__init__.py`](../../src/qie_research/encodings/__init__.py)
maps config-level string names to encoder classes, allowing the runner to
instantiate any encoding by name without conditional logic.

### 2. Config schema

A YAML config file at [`configs/smoke_test.yaml`](../../configs/smoke_test.yaml)
fully specifies one run: dataset, all three encodings with their
hyperparameters, downstream model, seed, and output directory.  The runner
is fully reproducible from the config file alone.

### 3. Config-driven runner

[`src/qie_research/runner.py`](../../src/qie_research/runner.py) reads a
config YAML and:

- Sets all random seeds before any data or model operation.
- Loads the specified dataset from a `DATASET_REGISTRY`.
- Splits into train and test using the config seed for reproducibility.
- For each encoding in the config list:
  - Instantiates the encoder from `ENCODING_REGISTRY`.
  - Times the encode step separately from the training step.
  - Tracks peak memory for encoding and training independently via `tracemalloc`.
  - Instantiates and trains the downstream model with the same seed.
  - Computes accuracy and F1-macro.
- Writes a structured JSON result file to the configured output directory.

Invoked as:

```bash
python -m qie_research.runner configs/smoke_test.yaml
```

### 4. Determinism smoke test

[`tests/test_determinism.py`](../../tests/test_determinism.py) runs the
runner twice on [`configs/smoke_test.yaml`](../../configs/smoke_test.yaml)
and asserts that metrics, output dimensions, encoding names, encoding params,
dataset split sizes, and seed are identical across both runs.

9 tests, all passing:

```
tests/test_determinism.py::test_config_exists                    PASSED
tests/test_determinism.py::test_same_number_of_encoding_results  PASSED
tests/test_determinism.py::test_same_encoding_names              PASSED
tests/test_determinism.py::test_same_encoding_params             PASSED
tests/test_determinism.py::test_same_output_dimensions           PASSED
tests/test_determinism.py::test_same_accuracy                    PASSED
tests/test_determinism.py::test_same_f1                          PASSED
tests/test_determinism.py::test_same_dataset_info                PASSED
tests/test_determinism.py::test_same_seed                        PASSED
```

### 5. Dependencies

[`requirements.txt`](../../requirements.txt) updated with exact pinned
versions of all dependencies.  [`pyproject.toml`](../../pyproject.toml)
added to register the package and allow editable install via `pip install -e .`.

## Smoke Run Results

Single run on UCI Wine, seed=42, logistic regression.  Full output at
[`results/metrics/smoke_test_wine.json`](../../results/metrics/smoke_test_wine.json):

| Encoding | Accuracy | F1 macro | d_out | Enc time | Train time |
|---|---|---|---|---|---|
| amplitude | 0.6667 | 0.5364 | 16 | 0.0008s | 0.0713s |
| angle | 0.9722 | 0.9710 | 26 | 0.0006s | 0.0677s |
| basis | 0.9167 | 0.9151 | 104 | 0.0008s | 0.0535s |

These numbers are infrastructure verification only and are not reportable
results.  No baseline comparison, no multiple seeds, and no geometric
diagnostics have been run.  The Phase 2 numerical audit must complete
before any performance claims are made.

## Verification Method

A teammate should verify that:

- `pytest tests/test_determinism.py -v` passes all 9 tests.
- `python -m qie_research.runner configs/smoke_test.yaml` completes without
  error and writes a JSON file to `results/metrics/`.
- The JSON output contains results for all three encodings.
- `pip install -r requirements.txt` followed by `pip install -e .` reproduces
  the environment from scratch.

After verification, the reviewer may change `Status` from `Ready for review`
to `Locked`.

## Artifact Paths

- Encoding implementations: [`src/qie_research/encodings/`](../../src/qie_research/encodings/)
- Config schema: [`configs/smoke_test.yaml`](../../configs/smoke_test.yaml)
- Runner: [`src/qie_research/runner.py`](../../src/qie_research/runner.py)
- Determinism test: [`tests/test_determinism.py`](../../tests/test_determinism.py)
- Smoke run output: [`results/metrics/smoke_test_wine.json`](../../results/metrics/smoke_test_wine.json)
- Dependencies: [`requirements.txt`](../../requirements.txt), [`pyproject.toml`](../../pyproject.toml)

## Completion Checklist

- [x] Config-driven runner built and functional.
- [x] Strict seed control enforced before any data or model operation.
- [x] Encoding time logged separately from training time.
- [x] Peak memory tracked for encoding and training independently.
- [x] Single-command execution for one dataset across all three encodings.
- [x] Determinism smoke test passing with 9/9 tests.
- [x] Dependencies pinned in [`requirements.txt`](../../requirements.txt).
- [x] Package installable via [`pyproject.toml`](../../pyproject.toml).

## Closure Statement

Phase 1 infrastructure is complete. The runner is deterministic, config-driven,
and executable by a teammate from a single command. Phase 2 numerical behavior
audit may begin.
