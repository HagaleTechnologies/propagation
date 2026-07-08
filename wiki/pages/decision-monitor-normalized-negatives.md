---
id: decision-monitor-normalized-negatives
title: Why does this repo use monitor-normalized negative sampling?
kind: decision-digest
status: current
maintainer: agent
sources:
  - ARCHITECTURE.md
  - docs/SPEC-labeling.md
verified:
  commit: 5c2dac7
  date: 2026-07-07
links:
  - overview
  - gotcha-eval-rules
---
A spot proves a path was open; the absence of a spot proves almost nothing —
nobody may have transmitted, or no receiver was listening. Naive labeling
("spot = 1, no spot = 0") trains a model that learns receiver geography, not
propagation. Monitor-normalized negative sampling is the chosen mitigation and
the novel methodological contribution targeted at TAPR/DCC and QEX publication.
The normative definition is in docs/SPEC-labeling.md §§2–4. ARCHITECTURE.md
§1.3 gives the design rationale.

## Digest

**Options considered:**
- Naive "no spot = closed": fast, simple, produces systematically biased labels.
  Learns which grid squares have active receivers.
- Random down-sampling of negatives: reduces class imbalance but does not
  address which negatives are trustworthy.
- **Monitor-normalized negatives (chosen)**: a negative is valid only when
  both sides were provably active — ≥1 receiver in RX was provably monitoring
  that band+mode_class in the window, AND ≥1 station in TX was provably
  transmitting. Windows failing either condition are unlabeled and excluded
  from both train and eval. This is a conservative, falsifiable negative.

**Key design choices (normative details in docs/SPEC-labeling.md):**
- Receiver-uptime evidence window: the labeled window padded ±30 min (§3).
  Mode_class-specific uptime (a CW-only skimmer cannot validate an FT8 negative).
- Transmit evidence: exact window, any receiver worldwide (§4.1).
- WSPR tier: beacon-scheduled transmissions are flagged separately —
  the "was anyone transmitting" condition is nearly free for WSPR.
- Training sampling: 3:1 negative:positive per (band, date) stratum;
  eval on the full unsampled set. See [[gotcha-eval-rules]].
- Monitor-count weighting (research-dependent): columns exist in labels
  Parquet to test it, but v1 is unweighted.

Receiver-uptime tables (`lake/receiver_uptime/`) are a first-class pipeline
artifact. Their quality directly controls eval validity. See ARCHITECTURE.md §1.3
and docs/SPEC-labeling.md §3 for the complete normative rules.
