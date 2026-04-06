# Peer Review Protocol

## Policy

Every change to the manuscript or codebase must be merged through a Pull Request reviewed by at least one teammate. This is mandatory, even for small wording fixes, unless the repository is temporarily inaccessible and the team has explicitly documented an exception in a repository record.

## Minimum Review Standard

Before merge, the Pull Request must have:

- One approval from a teammate who is not the author.
- No unresolved blocking comments.
- A linked repository record when the change has one.
- Updated documentation when workflow, claims, or structure changed.
- Evidence that tests, validations, or manuscript consistency checks were performed where relevant.

## Reviewer Responsibilities

Reviewers are expected to check:

- Scientific correctness and claim alignment.
- Reproducibility and config traceability.
- Fairness of matched-budget comparisons.
- Whether NTK, conditioning, or overhead claims are actually supported by artifacts.
- Whether changed figures, tables, or manuscript sentences remain synchronized.
- Whether reference updates follow the bibliography maintenance rules.

## Author Responsibilities

Authors must:

- Keep Pull Requests focused and reviewable.
- Respond to every substantive comment.
- Mark assumptions explicitly when evidence is still pending.
- Re-request review after major pushes.
- Avoid force-pushing after approval unless absolutely necessary; if done, announce what changed.

## Special Cases

### Manuscript Pull Requests

- Summarize the argument-level change, not just the file diff.
- Link every changed claim to the affected figure, table, or experiment record.
- If a result is provisional, label it clearly in both the Pull Request and manuscript text.

### Code Pull Requests

- State whether the change affects numerical results, runtime, determinism, or artifact format.
- Add or update tests when the logic could alter scientific conclusions.
- Call out any change that could invalidate previous results and open follow-up records if reruns are required.

### Protocol or Governance Pull Requests

- One approval is the minimum, but two approvals are strongly recommended because these files define team behavior.

## Merge Blocking Conditions

Do not merge if:

- The reviewer cannot reproduce the claimed effect from the linked artifacts.
- The Pull Request changes a headline result without updating manuscript language.
- Encoding overhead or data-handling implications are omitted from the discussion.
- The bibliography file was mass-reformatted for no scientific reason.
