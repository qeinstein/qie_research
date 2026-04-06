# Contributing Guide

This repository assumes three experienced Git users collaborating through a strict feature-branch workflow. All changes flow through GitHub Issues, branches, commits, Pull Requests, and review records. Direct commits to `main` are not permitted.

## Feature Branch Workflow

1. Open or claim a GitHub Issue before writing code, editing the manuscript, changing references, or updating protocols.
2. Create a branch from the latest `main` using a scoped name:
   - `code/<issue-id>-<short-topic>`
   - `text/<issue-id>-<section-name>`
   - `exp/<issue-id>-<dataset>-<encoding>`
   - `ref/<issue-id>-<citation-topic>`
   - `fix/<issue-id>-<problem>`
3. Keep the branch focused on one logical outcome. Split unrelated edits into separate branches and Pull Requests.
4. Push early and open a draft Pull Request once the work has a visible direction.
5. Rebase onto `main` before requesting final review so reviewers read the actual merge candidate.
6. Merge only after approval requirements, evidence requirements, and repository checks are satisfied.

## Required Pull Request Contents

Every Pull Request must include:

- A linked GitHub Issue.
- A concise statement of the scientific or operational objective.
- A summary of changed files and affected directories.
- A note describing how reproducibility was preserved.
- Artifact paths for any figures, tables, metrics, decision logs, or manuscript sections touched.
- Explicit mention of whether references, data manifests, or experiment outputs were modified.

## Review and Merge Rules

- Every manuscript or code change requires a Pull Request and approval from at least one teammate.
- Authors do not self-approve.
- Any substantial post-review push resets review expectations and should be re-reviewed.
- Unresolved blocking comments must be addressed before merge.
- Default merge method is `Rebase and merge` to keep a linear and inspectable scientific history.
- `Squash and merge` is acceptable for noisy exploratory branches only when the final commit message preserves the scientific rationale.

## Commit Discipline

- Follow the semantic commit scheme in [semantic_commit_guide.md](/home/fluxx/Workspace/qie_research/docs/governance/semantic_commit_guide.md).
- Reference the relevant issue number in the commit body when the issue is not already obvious from the branch name.
- Avoid mixing manuscript wording, code logic, and bibliography updates in the same commit unless they are inseparable.

## Manuscript Rules

- Keep manuscript source and draft material in `manuscript/`. Add format-specific subfolders later only if the team needs that split.
- Keep one section or one argument revision per Pull Request whenever possible.
- If a result table changes a claim, update the corresponding manuscript section, results artifact, and GitHub Issue in the same branch.

## Experiment and Data Rules

- Register planned runs in GitHub Issues before starting long experiments.
- Commit configuration, manifests, metadata, summary metrics, and publication-ready outputs.
- Do not commit raw datasets, caches, checkpoints, or heavyweight intermediate tensors.
- If encoding overhead exceeds training cost, document that explicitly in the linked issue and result summary.

## Reference Library Rules

- Follow [reference_library_maintenance.md](/home/fluxx/Workspace/qie_research/docs/governance/reference_library_maintenance.md) for every bibliography or PDF update.
- Bibliography-only changes should use a dedicated `ref/` branch unless the citation is inseparable from a manuscript claim.

## Branch Protection Expectations

Configure GitHub branch protection for `main` with at least:

- Required Pull Request before merge.
- At least one approving review.
- Dismiss stale approvals on new commits.
- Require conversation resolution before merge.
- Restrict force pushes and direct pushes.
