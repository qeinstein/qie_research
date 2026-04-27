# Missing Dependencies & Issues

Issues discovered during the RunPod sweep execution.

---

## 1. `ucimlrepo` not in `requirements-runpod.txt`

**Discovered:** During data preparation step on RunPod.

**Error:**
```
ModuleNotFoundError: No module named 'ucimlrepo'
```

**Affected:** `prepare_dry_bean.py`

**Fix:** Run `pip install ucimlrepo` on the pod before the sweep, or add it to `requirements-runpod.txt`.
