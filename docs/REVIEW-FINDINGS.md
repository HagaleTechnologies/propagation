# Adversarial Review Findings (2026-07-07)

A hostile-referee review of the full corpus. Five unambiguous defects were
fixed in place (CV-gap formula missing the horizon term; sample-weight use
made normative for objectives AND calibrators; ARCHITECTURE §7 wire-format
sketch marked superseded; lake layout aligned to store-once labels; dangling
cross-references). This file records what remains open — these are exactly
the objections a competent reviewer will raise, so they must survive into
the paper and the implementation.

## R1. Train/serve skew from spot-ingest latency — DECIDED

Training features use spot timestamps; the live system sees spots minutes
late, so a "trailing 15 min" feature differs materially at serving time.
Public archives lack ingestion timestamps; only the cqdx archive has
`ingestedAt`.

**Decision:** all autoregressive features are defined over spots with
`ts ≤ t_pred − Δ_avail`, with **Δ_avail = 5 min**, identically in training
and serving. This is source-agnostic (needs no ingestion timestamps) and
makes training features a faithful stand-in for serving-time features at the
cost of discarding the freshest 5 minutes. Validate Δ_avail empirically at
the serving milestone using cqdx `ingestedAt` (measure the actual ts→ingest
latency distribution); tighten or loosen then. SPEC-labeling's leakage rules
apply to the post-buffer feature definition.

## R2. Universe eligibility conditions on future information — OPEN (paper caveat)

Negative-eligibility for window W is only knowable after W closes; the
deployed nowcast must score cells whose labeled-population membership is
undefined at prediction time. This is an eval-vs-serving population shift.
The contract's abstain/sparse semantics mitigate the product side; the paper
needs an explicit deployment-population caveat in the limitations section.

## R3. Model exploits operator behavior — OPEN (framing constraint)

Behavioral targeting (operators transmit where they expect openings) affects
labels for both the ML model and the P.533 baseline equally — but the
learned model can additionally exploit operator behavior as *signal*, which
P.533 cannot. Defensible for a nowcasting product; the paper must claim
"predicts observed band openness," never "models ionospheric state."

## R4. k=1 positives with no SNR floor — ACCEPTED RISK

A single false FT8 decode can label a cell open. Callsign-hygiene filtering
mitigates; k=1 is a deliberate recall-favoring choice. Expect referee
probing; consider a k-sensitivity ablation in eval.

## R5. Schema `$id` hostname — DEFERRED

`prediction-surface.v1.schema.json` uses `github.com/thagale/...`; repos
elsewhere live under `HagaleTechnologies`. Fix when the remote is created.
