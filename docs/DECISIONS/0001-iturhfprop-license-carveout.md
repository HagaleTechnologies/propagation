# 0001: ITURHFProp vendored subtree is a licensing carve-out

## Status

Accepted.

## Context

M1 vendors ITURHFProp (the ITU-R P.533-14 reference implementation, C source
plus coefficient data) under `baselines/p533/upstream/`, so the P.533 baseline
comparison is reproducible by an outsider without cqdx access (README.md
"Licensing & boundaries"; ARCHITECTURE.md M-1). This repo as a whole is
intended to be open source (MIT OR Apache-2.0 dual license), but the upstream
ITU tree ships no LICENSE file — its terms are a narrower, implementer-scoped
grant, not an OSI-approved license. See `baselines/p533/PROVENANCE.md` for the
exact terms, where they're stated in `upstream/README.md`, and how they were
confirmed (no LICENSE/COPYING file, GitHub API reports `license: null`).

## Decision

Accept `baselines/p533/upstream/` as a licensing carve-out from this repo's
default license. The vendored C source and coefficient data under that path
retain the ITU's own implementer-grant terms; they are not relicensed to MIT
or Apache-2.0. The Python wrapper code (`src/propagation/models/p533*.py`)
that drives the vendored binary is ordinary repo code and stays under this
repo's normal dual license — only the vendored subtree itself carries
different terms.

## Consequences

- Anyone redistributing, forking, or building on `baselines/p533/upstream/`
  must honor the ITU's implementer-scoped terms (as-is, no warranty, granted
  to implementers of the Recommendation), not this repo's MIT/Apache-2.0
  license. The two license grants do not compose into a single blanket
  license for the repo.
- This carve-out must be disclosed at the top level (README.md), not only in
  `PROVENANCE.md`, before the repo's first public release — a downstream user
  scanning only the root license statement would otherwise miss it.
- If ITURHFProp is ever replaced or reimplemented from the Recommendation text
  directly (rather than vendored), this carve-out and its README callout
  should be removed in the same change.
