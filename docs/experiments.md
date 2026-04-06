# Experiment Planning

Use GitHub Issues, `configs/`, and this document for experiment planning records, not heavyweight outputs. Keep experiment planning simple until real files justify more structure.

Every experiment should map to a GitHub Issue and, when runnable, a config path under `configs/`.

## Minimum Requirements For Any Benchmark Family

- Use matched representational budget across QIE and classical comparators.
- Use matched optimization budget, matched regularization policy, and matched tuning effort.
- Include strong baselines, not only weak preprocessing baselines.
- Record predictive metrics and explanation-oriented diagnostics where relevant: NTK, CKA, effective rank, conditioning, and spectral decay.
- Track encoding wall-clock and memory overhead separately from model-training cost.
- Flag any case where encoding overhead is large enough to undermine practical usefulness.

The root execution plan is in [EXECUTION_PLAN.md](/home/fluxx/Workspace/qie_research/EXECUTION_PLAN.md). The focused benchmark rules are in [benchmark_execution_plan.md](/home/fluxx/Workspace/qie_research/docs/benchmark_execution_plan.md).
