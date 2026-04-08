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
- Mark assumptions explicitly when evidence is still pending.
- Re-request review after major pushes.


## Merge Blocking Conditions

Do not merge if:

- The reviewer cannot reproduce the claimed effect from the linked artifacts.
- The Pull Request changes a headline result without updating manuscript language.
- Encoding overhead or data-handling implications are omitted from the discussion.
