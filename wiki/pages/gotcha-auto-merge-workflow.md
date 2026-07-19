---
id: gotcha-auto-merge-workflow
title: What will bite you about this repo's auto-merge-on-open workflow?
kind: gotcha
status: current
maintainer: agent
sources:
  - .github/workflows/auto-merge-own-prs.yml
  - .github/workflows/ci.yml
verified:
  commit: 5431900
  date: 2026-07-19
links:
  - overview
---
`.github/workflows/auto-merge-own-prs.yml` enables `gh pr merge --auto
--squash` on any PR the repo owner opens. `--auto` merges as soon as
GitHub considers the PR mergeable — which, with no required status checks
configured, is immediately, before any CI even finishes running. M1's PR
#13 merged this way with two live-run-only bugs still present (see
[[gotcha-live-system-bugs]]), because nothing was gating it yet.

## Symptom

A PR merges within seconds of opening, before its own CI check has even
started, let alone finished — including a CI check that later turns out red.

## Cause and workaround

`gh pr merge --auto` only waits for checks that branch protection actually
marks as **required**; an unrequired check running in parallel doesn't block
it. The fix isn't touching this workflow — it's making sure `main`'s branch
protection always has a required status check (currently `test`, added
after this bit once) AND requires changes to go through a PR at all
(added in the same pass — the first branch-protection setup only added the
required check, not the PR requirement, which would have still allowed a
direct `git push origin main` bypassing CI entirely). Check both are still
set before trusting this workflow: `gh api
repos/HagaleTechnologies/propagation/branches/main/protection`.
