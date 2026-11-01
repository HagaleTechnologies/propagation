---
id: gotcha-codex-gate-and-dependabot
title: What will bite you about the Codex gate and Dependabot PRs?
kind: gotcha
status: current
maintainer: agent
sources:
  - .github/workflows/wait-for-codex.yml
  - .github/workflows/ci.yml
  - tests/test_wait_for_codex_dependabot.py
verified:
  commit: 11a3818
  date: 2026-11-01
links:
  - overview
---
The required check `Codex review clean (gate before expensive downstream
jobs)` exists on a commit only when a job in `wait-for-codex.yml` writes it
through the Checks API. No job carries that name, so skipping a job does not
satisfy it. Dependabot updates get it from the producer job's pass-through
(PRO-18). Three things here are easy to get wrong.

## Symptom

- A Dependabot PR has green `test` and `Wiki lint` but never becomes
  mergeable, and the gate check is missing (not failed, not pending). Before
  PRO-18 this happened to every Dependabot PR: PR #73 merged only because
  someone posted `@codex review` by hand.
- A comment or review finding says "GitHub withholds Actions secrets from
  Dependabot-triggered runs, so this job cannot run for Dependabot." That is
  true for `pull_request` and false for `pull_request_target`. The claim was
  copied between repos and workflow files for months without a live check.
- An edit to `wait-for-codex.yml` seems to have no effect on its own PR.

## Cause and workaround

- **Gate check written explicitly.** Since #70 the producer job's name differs
  from the required check's name. Any path that should pass a PR must call
  `upsert_check_run`. Excluding an actor in a job-level `if:` removes the
  check entirely. Route new pass-through cases through `create-check`.
- **Secrets for Dependabot.** A Dependabot-triggered `pull_request` run gets
  `Secret source: Dependabot` and a read-only token. A Dependabot-triggered
  `pull_request_target` run gets `Secret source: Actions`, including the
  main-only `codex-review` environment's secrets. The evidence is in the
  producer job's header comment in `wait-for-codex.yml`. Check the "Set up
  job" log before trusting either claim.
- **Base-branch copy.** `pull_request_target` runs `main`'s copy of the
  workflow, so a PR cannot exercise its own change. `ci.yml`'s `test` job runs
  `tests/test_wait_for_codex_dependabot.py` on the PR's own copy. It executes
  `create-check`'s script with a stub `gh` and is the only pre-merge check of
  the pass-through.
- `ci.yml`'s `codex-clean` job must keep skipping `dependabot[bot]`. The
  pass-through applies no `codex-clean:<sha12>` label, so polling would time
  out on a Dependabot rebase.
