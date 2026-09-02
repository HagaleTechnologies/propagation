# TAPR/DCC writeup — draft outline

Status: draft outline only (PRO-12). Not a paper draft. Target venue: TAPR/DCC
proceedings (ARCHITECTURE.md §9); QEX (ARRL) as a secondary/adapted target.

## Working title

"Learned HF band-openness nowcasting vs. ITU P.533: an amateur-radio-spot
benchmark" (placeholder — bikeshed later)

## 1. Problem statement

- HF propagation prediction today means running VOACAP/ITU P.533 point
  estimates from smoothed indices — no calibrated probability, no use of the
  huge volume of real-time spot data amateur radio already produces
  (WSPRnet/PSKReporter/RBN).
- Question: does a model trained directly on observed spot outcomes beat
  P.533 at short-horizon band-openness nowcasting, using only data an
  amateur station can already see?

## 2. Data & method

- Sources: WSPRnet (batch archive, primary), RBN (CW ground truth, PRO-8),
  live PSKReporter MQTT (PRO-9). cqdx R2 archive as an optional private
  accelerant (PRO-10) — public-only pipeline is the one being claimed.
- Label construction: monitor-normalized openness/SNR from spot density,
  not raw spot counts (observation-bias correction — this is itself a
  contribution per ARCHITECTURE.md §9).
- Feature set: solar geometry, space-weather (OMNI2), grid-cell path
  history, receiver-uptime normalization.
- Model: LightGBM (GBT), climatology and ITU P.533 as baselines.
- Evaluation: blocked time-series CV, train 2024-01/02/03, eval
  2024-05/07/09 (2024-05 spans the May 2024 Gannon storm). Brier score and
  log-loss, proper scoring rules.

## 3. Headline result

- ADR 0006: GBT beats both climatology and P.533 on Brier at h=0 and h=+3h
  on 20m/15m/10m, all three eval months, by a wide margin (climatology
  ~0.08–0.11, P.533 ~0.31–0.40, GBT ~0.05–0.08 across bands/horizons).
- Include the ADR 0006 headline table verbatim (or a formatted version of
  it) as the paper's central figure.
- Storm-window behavior (PRO-11 case studies) as a secondary figure — where
  the model should win biggest per ARCHITECTURE.md §6.

## 4. Limitations

- Per-band models fit independently in the ADR 0006 run, not the single
  shared-across-bands model the architecture intends (§5) — note this
  explicitly if the shared-model property is ever implied.
- Public-only claim depends on PRO-10 landing: the headline number must be
  reproducible from WSPRnet/RBN/PSKReporter alone before this section can
  say "no private data required."
- P.533 baseline uses standard climatological indices, not the operator's
  live conditions — acknowledge this is the fairest baseline available, not
  a strawman, but still a different information regime than the GBT.
- Amateur spot networks have their own biases (operator density, antenna/
  power skew) — the monitor-normalization method mitigates but doesn't
  eliminate this.

## 5. Next steps / future work

- M4: live serving (`serving/score.py`, `prediction-surface.v1` contract),
  continuous scoreboard vs. reality (ARCHITECTURE.md §7).
- Full storm-window case study set (2-3 storms, PRO-11).
- Cross-band shared model (architecture's original §5 intent) as a
  follow-up result once M3's per-band bar is fully closed.
- Longer eval horizon / more storm seasons once the live PSKReporter feed
  (PRO-9) has accumulated enough history.

## Open questions before drafting begins

- Co-authorship / attribution.
- Whether to lead with the dataset-construction contribution (§9) or the
  benchmark result — may differ between TAPR/DCC (audience skews method +
  dataset) and QEX (audience skews "does this help me on the air").
- Figure/table budget for DCC's typical paper length.
