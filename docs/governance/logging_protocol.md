# Repository Record Logging Protocol

## Purpose

Repository records are the authoritative operational log for this project. Every milestone, phase outcome, experiment family, meeting decision, and blocking problem must be traceable through a Markdown record under `records/`.

Local scratch notes are allowed during work, but a record committed under `records/` is the only project memory assumed to persist.

## Mandatory Record Categories

- `phase-outcome`: Used to close a planned research phase with evidence. Store under `records/phases/`.
- `experiment-run`: Used to register, monitor, and summarize experimental runs or run families. Store under `records/experiments/`.
- `meeting-note`: Used to capture agendas, decisions, and action items from team meetings. Store under `records/meetings/`.
- `decision`: Used when methodology, datasets, narrative framing, or tooling policy changes. Store under `records/decisions/`.
- `risk`: Used when scientific, computational, or operational risks need explicit mitigation. Store under `records/risks/`.

Copy templates from `records/templates/` when creating new records.

## Required Metadata

Every record must contain:

- One clear owner.
- One status.
- One related phase when the record belongs to the formal execution plan.
- One record type.
- A short problem statement or outcome target.
- Explicit acceptance criteria or completion checklist.
- Linked branch and Pull Request once work begins, when a Pull Request exists.
- Artifact paths for generated evidence, if any.

## Title And Filename Conventions

Use one of the following title patterns:

- `[PHASE 0] Scope lock and narrative freeze`
- `[PHASE 2] Encoding numerics verified before training`
- `[EXP][fashion-mnist][angle] matched-budget sweep`
- `[EXP][synthetic-high-rank-noise][amplitude] NTK stress test`
- `[MEETING][2026-01-29] weekly synchronization`
- `[DECISION][2026-02-03] dataset freeze criteria`
- `[RISK][2026-02-10] encoding overhead exceeds training budget`

Use one of the following filename patterns:

- `records/phases/phase-0-scope-lock.md`
- `records/experiments/2026-02-03-fashion-mnist-angle-sweep.md`
- `records/meetings/2026-01-29-weekly-synchronization.md`
- `records/decisions/2026-02-03-dataset-freeze-criteria.md`
- `records/risks/2026-02-10-encoding-overhead.md`

## Record Body Requirements

### For `phase-outcome`

- Phase objective.
- Inputs reviewed.
- Required artifact paths.
- Verification method.
- Reviewer or approver.
- Close statement summarizing whether the outcome is positive, equivalent, negative, or not yet determined.

### For `experiment-run`

- Dataset and split definition.
- Encoding or baseline under test.
- Matched representational budget.
- Seed plan.
- Config path.
- Expected outputs.
- Runtime and overhead tracking plan.

### For `meeting-note`

- Attendees.
- Agenda.
- Decisions made.
- Action items with owners and due dates.
- Follow-up records or Pull Requests created.

## Lifecycle Rules

1. Create the record before work starts when the work affects a phase outcome, experiment, decision, meeting, or risk.
2. Keep the main body updated rather than scattering status across disconnected notes.
3. Use dated entries only for timestamped milestones, review requests, or final evidence.
4. Link every merged Pull Request back to the relevant record.
5. Close the record only when acceptance criteria and evidence are satisfied.

## Phase Outcome Closure Standard

A phase outcome record cannot be closed unless it contains:

- Linked Pull Request or commit range.
- Final artifact paths in the repository.
- Verification note naming who checked the result.
- Statement of whether the phase supports the geometric-advantage, numerical-equivalence, negative-result, or not-yet-determined narrative.

## Experiment Logging Standard

For long-running or many-seed experiments, one parent record may track the run family and child records may track exceptional failures, ablations, or reruns. Each final result table or figure used in the manuscript must point back to a single authoritative experiment record.

## Prohibited Practices

- Closing records without evidence.
- Combining unrelated outcomes in one record.
- Logging experiments only in commit messages or chat.
- Changing dataset scope, architecture scope, or benchmark budget without a decision record.
