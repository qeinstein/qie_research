# GitHub Issue Logging Protocol

## Purpose

GitHub Issues are the authoritative operational log for this project. Every milestone, phase outcome, experiment family, meeting decision, and blocking problem must be traceable through an issue. Local notes are allowed during work, but the issue record is the only project memory assumed to persist.

## Mandatory Issue Categories

- `phase-outcome`: Used to close a planned research phase with evidence.
- `experiment-run`: Used to register, monitor, and summarize experimental runs or run families.
- `meeting-note`: Used to capture agendas, decisions, and action items from team meetings.
- `decision`: Used when methodology, datasets, narrative framing, or tooling policy changes.
- `risk`: Used when scientific, computational, or operational risks need explicit mitigation.

## Required Metadata

Every issue must contain:

- One clear owner.
- One milestone.
- At least one `phase:*` label when the issue belongs to the formal execution plan.
- One `type:*` label describing the issue category.
- A short problem statement or outcome target.
- Explicit acceptance criteria.
- Linked branch and Pull Request once work begins.
- Artifact paths for generated evidence, if any.

## Title Conventions

Use one of the following title patterns:

- `[PHASE 0] Scope lock and narrative freeze`
- `[PHASE 2] Encoding numerics verified before training`
- `[EXP][fashion-mnist][angle] matched-budget sweep`
- `[EXP][synthetic-high-rank-noise][amplitude] NTK stress test`
- `[MEETING][2026-01-29] weekly synchronization`
- `[DECISION][2026-02-03] dataset freeze criteria`
- `[RISK][2026-02-10] encoding overhead exceeds training budget`

## Issue Body Requirements

### For `phase-outcome`

- Phase objective.
- Inputs reviewed.
- Required artifact paths.
- Verification method.
- Reviewer or approver.
- Close statement summarizing whether the outcome is positive, equivalent, or negative.

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
- Follow-up issues or Pull Requests created.

## Lifecycle Rules

1. Open the issue before work starts.
2. Keep the top comment updated rather than scattering status across many comments.
3. Use comments only for timestamped milestones, review requests, or final evidence.
4. Link every merged Pull Request back to the issue.
5. Close the issue only when acceptance criteria and evidence are satisfied.

## Phase Outcome Closure Standard

A phase outcome issue cannot be closed unless it contains:

- Linked Pull Request or commit range.
- Final artifact paths in the repository.
- Verification note naming who checked the result.
- Statement of whether the phase supports the geometric-advantage, numerical-equivalence, or negative-result narrative.

## Experiment Logging Standard

For long-running or many-seed experiments, one parent issue may track the run family and child issues may track exceptional failures, ablations, or reruns. Each final result table or figure used in the manuscript must point back to a single authoritative experiment issue.

## Prohibited Practices

- Closing issues without evidence.
- Combining unrelated outcomes in one issue.
- Logging experiments only in commit messages or chat.
- Changing dataset scope, architecture scope, or benchmark budget without a decision issue.
