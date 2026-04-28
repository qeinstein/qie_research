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

---

## 2. Covertype seed 42 ran on full 465k samples (inconsistent with seeds 1337–99)

**Discovered:** During the first full sweep on RunPod (2026-04-27).

**Problem:** Seed 42 of covertype ran before `max_samples: 50000` was added to
`configs/covertype.yaml`. Seeds 1337, 2026, 7, and 99 ran on 50k samples.
Mixing these in a mean±std computation across seeds is methodologically invalid.

**Fix:** Rerun only covertype seed 42 under the 50k cap and overwrite just that
one result file:

```bash
python3 -m qie_research.runner configs/covertype.yaml --seed 42
```

This will overwrite `results/metrics/covertype/covertype_seed42.json` with the
50k-capped result, making all 5 seeds consistent. Run this before any analysis.
