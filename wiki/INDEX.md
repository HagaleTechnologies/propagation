# propagation wiki index

- [propagation — what is this and where do things live?](pages/overview.md) — ML-based HF propagation nowcasting: train on real path-report data
- [How does the data pipeline work and why is it pluggable?](pages/subsystem-data-pipeline.md) — The data pipeline converts heterogeneous amateur-radio spot archives from
- [Why does this repo use monitor-normalized negative sampling?](pages/decision-monitor-normalized-negatives.md) — A spot proves a path was open; the absence of a spot proves almost nothing —
- [Why is ITURHFProp vendored here rather than calling cqdx's P.533 sidecar?](pages/decision-p533-baseline.md) — The P.533 baseline is implemented by vendoring the ITU's public ITURHFProp C
- [Why is the prediction surface columnar and integer-quantized?](pages/decision-prediction-contract.md) — The prediction surface contract (v1) uses columnar parallel arrays (not
- [What will bite you about evaluation (blocked CV, full label set)?](pages/gotcha-eval-rules.md) — Three eval rules will produce silently invalid results if violated: (1) random
- [What will bite you if you add a cqdx dependency?](pages/gotcha-open-closed-boundary.md) — This repo is intended to be open source; cqdx is closed. If you import any
- [What contract does this repo expose for HF band-opening predictions?](pages/interface-prediction-surface.md) — This repo PRODUCES the `propagation.prediction-surface.v1` contract — a
