# Repository Records

This directory is the authoritative operational log for the project. Use these Markdown records for phase outcomes, experiment registration, meeting notes, decisions, and risk tracking.

Repository records are versioned with the rest of the project, reviewed in Pull Requests, and linked from manuscripts, result summaries, and governance changes when they affect scientific claims.

## Record Categories

- `records/phases/`: closure records for formal execution-plan phases.
- `records/experiments/`: planned runs, sweeps, ablations, reruns, and final experiment summaries.
- `records/decisions/`: methodology, dataset, narrative, tooling, or scope changes.
- `records/meetings/`: agendas, decisions, and action items from team meetings.
- `records/risks/`: scientific, computational, operational, and reproducibility risks.
- `records/templates/`: copyable templates for each record type.

## Naming Conventions

- Phase records: `phase-N-short-name.md`, for example `phase-0-scope-lock.md`.
- Experiment records: `YYYY-MM-DD-dataset-method-purpose.md`.
- Decision records: `YYYY-MM-DD-short-decision.md`.
- Meeting records: `YYYY-MM-DD-meeting-note.md`.
- Risk records: `YYYY-MM-DD-short-risk.md`.

## Completion Standard

A record is not closed until it includes the relevant artifact paths, verification method, reviewer or approver, linked Pull Request or commit range, and a clear outcome statement.
