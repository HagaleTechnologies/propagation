# Contributing

This repo takes contributions, but read `README.md` and `ARCHITECTURE.md`
first — the open/closed boundary with cqdx (a separate closed product) is
deliberate and enforced: no cqdx code imports, no cqdx-internal assumptions,
and everything here must be reproducible from public data sources.

## Setup

```bash
uv sync
uv run build-p533   # builds the vendored ITU-R P.533 baseline (baselines/p533/)
uv run pytest -q
uv run ruff check .
```

Python 3.11+, managed with [uv](https://github.com/astral-sh/uv).

## Before opening a PR

- `uv run pytest -q` and `uv run ruff check .` must both be clean — CI
  enforces this on every PR.
- Follow the labeling methodology in `docs/SPEC-labeling.md` exactly for
  anything touching label construction; it's designed so two independent
  implementations produce identical label sets.
- Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- Read `wiki/INDEX.md` for accumulated gotchas/decisions before deep-diving
  into a subsystem; `docs/DECISIONS/` holds the normative ADRs.

## Licensing

Dual-licensed MIT OR Apache-2.0 (see `LICENSE-MIT` / `LICENSE-APACHE`), except
`baselines/p533/upstream/` (vendored ITU-R P.533 reference source), which
carries the ITU's own implementer-scoped grant — see
`docs/DECISIONS/0001-iturhfprop-license-carveout.md`. By contributing, you
agree your contributions are licensed under the same dual license as the
rest of the repo.
