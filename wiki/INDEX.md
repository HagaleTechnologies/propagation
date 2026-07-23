# propagation wiki index

- [propagation — what is this and where do things live?](pages/overview.md) — ML-based HF propagation nowcasting: train on real path-report data
- [How does the data pipeline work and why is it pluggable?](pages/subsystem-data-pipeline.md) — The data pipeline converts heterogeneous amateur-radio spot archives from
- [Which M0 scope reductions were deliberate, and when do they expire?](pages/decision-m0-scope-reductions.md) — The M0 implementation (PR #11) contains four deliberate, user-approved scope
- [Why did M2 start despite M1 not beating M0 on the eval harness?](pages/decision-m1-ladder-exception.md) — ARCHITECTURE.md §5 says the modeling ladder is "strictly ordered; each rung
- [Why does this repo use monitor-normalized negative sampling?](pages/decision-monitor-normalized-negatives.md) — A spot proves a path was open; the absence of a spot proves almost nothing —
- [Why is ITURHFProp vendored here rather than calling cqdx's P.533 sidecar?](pages/decision-p533-baseline.md) — The P.533 baseline is implemented by vendoring the ITU's public ITURHFProp C
- [Why is the prediction surface columnar and integer-quantized?](pages/decision-prediction-contract.md) — The prediction surface contract (v1) uses columnar parallel arrays (not
- [What will bite you about this repo's auto-merge-on-open workflow?](pages/gotcha-auto-merge-workflow.md) — `.github/workflows/auto-merge-own-prs.yml` enables `gh pr merge --auto
- [What will bite you about evaluation (blocked CV, full label set)?](pages/gotcha-eval-rules.md) — Three eval rules will produce silently invalid results if violated: (1) random
- [What will bite you about a leakage audit that only blocklists columns by name?](pages/gotcha-label-leakage-via-nulls.md) — M2's leakage audit (`tests/test_leakage.py`) blocklisted label columns by
- [What will bite you if you only test against mocked externals/binaries?](pages/gotcha-live-system-bugs.md) — M1 shipped through nine task-level reviews and a whole-branch review with a
- [What will bite you if you add a cqdx dependency?](pages/gotcha-open-closed-boundary.md) — This repo is intended to be open source; cqdx is closed. If you import any
- [What will bite you about implementing from a plan drafted before its dependencies merged?](pages/gotcha-plan-drift-before-merge.md) — A single PR (#10) once drafted implementation plans for M1 through M4, all
- [What will bite you about the QA diurnal-ratio checks (1 and 2)?](pages/gotcha-qa-diurnal-checks.md) — QA checks 1 (20m day/night open-rate ratio > 2) and 2 (160m/80m night/day
- [What contract does this repo expose for HF band-opening predictions?](pages/interface-prediction-surface.md) — This repo PRODUCES the `propagation.prediction-surface.v1` contract — a
