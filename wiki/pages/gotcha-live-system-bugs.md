---
id: gotcha-live-system-bugs
title: What will bite you if you only test against mocked externals/binaries?
kind: gotcha
status: current
maintainer: agent
sources:
  - src/propagation/models/p533.py
  - src/propagation/eval/stratify.py
verified:
  commit: 5431900
  date: 2026-07-19
links:
  - decision-p533-baseline
  - overview
---
M1 shipped through nine task-level reviews and a whole-branch review with a
full green test suite — three real, independent bugs still slipped through,
each caught only by an actual live acceptance run against the real WSPRnet
archives, the real vendored ITURHFProp binary, and the real GFZ/SWPC
endpoints. If your code's test suite mocks every external boundary (a
subprocess, an HTTP call, a vendored binary), plan for a live run before
trusting it, not instead of one.

## Symptom

- A physically-plausible-looking config value (a transmit power, a unit
  conversion) that's silently wrong by an order of magnitude — passes every
  mocked test because no test asserts the real numeric value end-to-end.
- An HTTP fetch that "succeeds" (200-equivalent, no exception) but silently
  returns the wrong content, because the mocked test never exercises the
  real server's actual behavior (a redirect, a moved endpoint).
- A vendored binary that segfaults on a valid-looking input the Python
  wrapper never told it was invalid — mocked subprocess tests can't discover
  what the real binary actually does with edge-case input.

## Cause and workaround

Three concrete instances from M1, each fixed only after a real live run
(see the fix commits, not restated here): a transmit-power unit mismatch
(linear vs. logarithmic) in `p533.py`'s input-card renderer; a Kp-archive
fetch that followed a real HTTP redirect incorrectly in `stratify.py`; and
a vendored-binary crash on a frequency outside its physically valid range,
also in `p533.py`. None were reachable from any mocked test — all three
needed the real network endpoint or the real compiled binary running.

Workaround: after implementation + code review passes, budget time for an
actual live run against real external systems (real archives downloaded,
real binaries executed, real HTTP endpoints hit) before calling a milestone
touching external data or vendored binaries complete. Treat a clean mocked
test suite as necessary, not sufficient.
