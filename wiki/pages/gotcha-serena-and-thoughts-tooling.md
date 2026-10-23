---
id: gotcha-serena-and-thoughts-tooling
title: What will bite you about Serena symbol search and thoughts docs?
kind: gotcha
status: current
maintainer: agent
sources:
  - .serena/project.yml
  - tests/test_serena_project_config.py
  - .gitignore
verified:
  commit: 5a84c96
  date: 2026-10-23
links:
  - overview
---
Serena symbol search in this repo is configured by `.serena/project.yml`.
Serena reads that file when an agent calls `activate_project` (PRO-13). It
indexes `python` only, is `read_only`, and skips the vendored ITURHFProp tree.
Five things about it are easy to get wrong, plus one about thoughts docs.

## Symptom

- After a Serena upgrade, the comments in `project.yml` disappear and fields
  get reordered or appended. Serena rewrote the file because a field was
  missing or had a pre-2.0 name. `tests/test_serena_project_config.py` fails
  on the key set when this happens.
- `serena project is_ignored_path data/x.py` reports "not ignored" even though
  `.gitignore` has `/data/`. Only the diagnostic is wrong: the index walker
  skips `data/` and `reports/`. The diagnostic also reports "not ignored" for
  any path that does not exist yet.
- `activate_project` fails with `session_id Field required` on Serena 2.x.
- `uv tool install --from git+https://github.com/oraios/serena@…` fails inside
  Catalyst runner containers.
- In cloud sessions the codebase-* agents still grep instead of using Serena.

## Cause and workaround

- Upgrades: list the new version's fields with
  `dataclasses.fields(ProjectConfig)` and update `project.yml` and the test's
  `EXPECTED_KEYS` together. The header comment in `project.yml` names the
  pinned version.
- Ignore checks: to prove a directory is skipped, plant a `.py` file there and
  confirm `serena project index .` reports the same file count.
- Activation: call `initial_instructions` first and pass the session id it
  prints.
- Install: the runner sets `GIT_CONFIG_COUNT`, which breaks uv's git clone.
  Prefix the install with `GIT_CONFIG_COUNT=0`.
- Cloud sessions register no MCP servers, so no commit here can give them
  Serena. Check agent-side activation from a laptop session that has the
  Serena MCP server registered.
- Thoughts docs: `thoughts/` is a gitignored symlink into the central thoughts
  repo. Container phase runs reach it through the runner's own commit and
  push, and local clones reach it through `humanlayer thoughts sync`. On
  2026-10-23 `repos/propagation/` in that repo was fully tracked and clean.
