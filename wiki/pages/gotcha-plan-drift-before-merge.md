---
id: gotcha-plan-drift-before-merge
title: What will bite you about implementing from a plan drafted before its dependencies merged?
kind: gotcha
status: current
maintainer: agent
sources:
  - docs/superpowers/plans/2026-07-15-m1-p533-baseline.md
verified:
  commit: 5431900
  date: 2026-07-19
links:
  - decision-p533-baseline
  - overview
---
A single PR (#10) once drafted implementation plans for M1 through M4, all
before M0 had actually merged. By the time M1 planning started, that draft's
"interface assumptions" had diverged substantially from the real M0 code —
`Lake` class, `OpennessModel` protocol, `predict_p_open()`, a dict-based
`headline_table()`, an ordered `BANDS` list — none of which exist in the
merged code. Every one would have produced silently-wrong implementation
work if trusted at face value.

## Symptom

A plan (or design doc, or interface spec) written against code that didn't
exist yet, consulted after that code actually merged with different names,
shapes, or behavior than assumed. The plan reads as confident and specific;
nothing in it flags that it's stale.

## Cause and workaround

Plans and specs are frozen at write-time; merged code is a moving target
until it lands. A plan drafted across a milestone boundary (referencing
interfaces from work not yet merged) is a claim about what that code will
look like, not what it does look like.

Workaround: before executing any plan that assumes interfaces from a
different, already-merged milestone, reconcile every consumed interface
against the actual current code first — grep for the real class/function
names, read the real signatures, don't trust the plan's code samples. M1's
plan was rewritten from scratch this way (harvesting what survived contact
with the real M0 code, discarding what didn't) rather than patched in place;
PR #10's M2–M4 sections should get the same treatment, not be trusted, when
M2/M3/M4 planning starts.
