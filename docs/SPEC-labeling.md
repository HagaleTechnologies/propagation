# SPEC: Label construction (v1)

Status: v1.0 (2026-07-06). Normative. Two independent implementations following
this spec MUST produce identical label sets from identical input Parquet.
Deviations from ARCHITECTURE.md are marked **[deviation]** with rationale.

Notation: window `W = [t0, t0+15min)` aligned to UTC 15-minute boundaries
(`t0.minute ∈ {0,15,30,45}`, seconds zero). A spot belongs to the window
containing its `ts` (floor). Cells are directional:
`(tx_field, rx_field, band)`; `EM→PM ≠ PM→EM`.

---

## 1. Spot hygiene (applied before everything else)

A spot row is **qualifying** iff all of:

1. `ts`, `band`, `mode`, `dx_call`, `de_call` non-null; `band` in the supported
   set (`160m 80m 60m 40m 30m 20m 17m 15m 12m 10m 6m`).
2. Both sides have a usable location:
   - Grid path: take the reported grid, uppercase, truncate to 4 chars if
     longer. Valid iff chars 1–2 ∈ `A..R` and chars 3–4 ∈ `0..9`. A bare
     2-char field (chars 1–2 ∈ `A..R`, nothing else) is accepted with
     `grid4 = NULL` (usable at field granularity only).
   - Blocklist: reject the literal grid `RR73` (FT8 message artifact leaking
     into reports). Count rejections (§6.8).
   - No grid but lat/lon present: derive the field/grid4 from lat/lon.
   - Neither: the spot is non-qualifying.
3. Callsign filters: reject if either call matches `/MM` or `/AM` suffix
   (maritime/aeronautical mobile — location unstable). `/P`, `/M`, `/QRP` and
   other suffixes are kept (the per-report grid governs). Reject calls that
   are not plausible callsigns: must match
   `^[A-Z0-9]{1,3}[0-9][A-Z0-9]{0,3}[A-Z](/[A-Z0-9]{1,4})?$` after stripping a
   leading `<`/trailing `>` (hashed-call markers).
4. Self-spots (`dx_call == de_call` after suffix stripping) are rejected.
5. Implied great-circle distance ≥ 25 km (kills same-station artifacts).

`mode_class` is derived from `mode`:

| mode_class | modes |
|---|---|
| `digi` | FT8, FT4, WSPR, FST4, FST4W, JS8, JT65, JT9, Q65, MSK144 |
| `cw` | CW, RTTY (RBN skimmer modes) |
| `other` | everything else (SSB, human cluster spots, …) |

**Cross-source dedup.** Dedup key: `(dx_call, de_call, band, mode, window)`.
Keep exactly one row per key, chosen by source priority
`wsprnet > rbn > pskreporter > cqdx`, tie-break: highest non-null `snr_db`,
then lowest `ts`. Applied before labeling AND before any spot-count feature.

## 2. Universe definition

The label universe is **activity-gated**, not the full 324×324×bands cross
product (which is >99% permanently-empty ocean/ocean pairs and would swamp
every count with structural zeros).

For window `W` and band `b`, the universe contains cell `(TX, RX, b, W)` iff
**either**:

- **(P)** ≥1 qualifying spot with `tx_field=TX, rx_field=RX, band=b, ts ∈ W`; or
- **(N-eligible)** ∃ a mode_class `m ∈ {digi, cw}` such that both:
  - **monitor condition**: ≥1 receiver located in `RX` provably monitoring
    `(b, m)` in `W` (§3), and
  - **tx condition**: ≥1 station located in `TX` provably transmitting on
    `(b, m)` in `W` (§4.1).

`other` mode_class never establishes N-eligibility (human spotting proves
nothing about continuous monitoring) but its spots DO count as positives.

Everything outside the universe is **unlabeled** and excluded from train and
eval. Report the universe size and the unlabeled-activity fraction per
band/day (§6.8).

## 3. Receiver-uptime tables

**Definition.** Receiver `R` (identity = `de_call` as reported, post-hygiene)
is *provably monitoring* `(band b, mode_class m)` in window `W = [t0, t0+15m)`
iff `R` produced ≥1 qualifying spot with that band and a mode in `m` with
`ts ∈ [t0 − 30min, t0 + 45min)`.

- The evidence interval is the window padded by 30 min on both sides
  (window end `t0+15m` + 30 min = `t0+45m`). This IS the hysteresis/hang:
  a receiver that decodes something every ≤75 min is continuously "up". No
  additional decay logic. **[deviation — formalization]** ARCHITECTURE §1.3
  says "[t−30min, t+30min]"; this spec pins `t` to the padded window so the
  interval is unambiguous.
- Per `(receiver, band, mode_class)` independently. Decoding on 20m proves
  nothing about 15m; a CW skimmer proves nothing about FT8.
  **[deviation — refinement]** ARCHITECTURE keys uptime on (receiver, band)
  only. A CW-only skimmer cannot hear an FT8 transmitter, so band-level
  uptime produces false negatives; mode_class matching (§4) fixes this.
- Receiver location = the modal `de_grid4` (or field, if only fields reported)
  across its reports in the evidence interval; ties broken lexicographically.
  Receivers with no usable location contribute nothing.

**Storage** (`lake/receiver_uptime/band=…/date=…/*.parquet`):

| column | type |
|---|---|
| window_start | timestamp (UTC) |
| de_call | utf8 |
| de_field | utf8(2) |
| de_grid4 | utf8(4), nullable |
| band | utf8 |
| mode_class | utf8 (`digi`\|`cw`) |
| n_evidence_reports | int32 |
| first_evidence_ts / last_evidence_ts | timestamp |

## 4. Labels

### 4.1 Transmit-activity evidence

Station `S` is *provably transmitting* on `(b, m)` in `W` iff ≥1 qualifying
spot exists with `dx_call = S`, band `b`, mode ∈ `m`, and `ts ∈ W` (the exact
window, no padding — a spot timestamps an actual transmission), from ANY
receiver worldwide. Same band required; being spotted on 40m is not evidence
of transmitting on 20m. `S`'s location = the `tx_field` of that spot.

WSPR tier: WSPR spots additionally carry `tx_dbm`; WSPR-derived tx evidence is
flagged `evidence_tier = 'wspr'` (beacon-scheduled, calibrated) vs `'spot'`.

### 4.2 Positive label

`open(TX, RX, b, W) = 1` iff ≥1 qualifying (post-dedup) spot with
`tx_field=TX, rx_field=RX, band=b, ts ∈ W`. Any mode. `k=1` distinct
`(dx_call, de_call)` pair suffices (per ARCHITECTURE §1.2). No SNR floor —
a decode is a decode; the mode's decode threshold is the floor.

### 4.3 Negative label

`open(TX, RX, b, W) = 0` iff the cell is N-eligible (§2) via at least one
mode_class AND zero qualifying spots `TX→RX` on band `b` in `W` **in any
mode**. (Monitor and tx evidence must share a mode_class; the "no spot"
condition is mode-agnostic.)

Recorded per negative: `n_monitors` (distinct monitoring receivers in RX,
summed over eligible mode_classes), `n_tx_stations`, `evidence_tier`
(`wspr` if any tx evidence is WSPR, else `spot`). Unweighted in v1
(ARCHITECTURE flags monitor-count weighting as research-dependent; the
columns exist so it costs nothing to test).

### 4.4 SNR target (secondary)

Defined only where `open=1` and ≥1 contributing spot has non-null SNR.
Target = median over contributing spots of **FT8-equivalent SNR**:

```
snr_ft8eq = snr_reported + bw_offset(mode) + pwr_offset
bw_offset:  FT8/FT4/WSPR/FST4/JS8/JT65/JT9/Q65 → 0    (reported re 2500 Hz)
            CW/RTTY via RBN                    → −7    (re 500 Hz; −10·log10(2500/500) ≈ −7.0, use −7)
            other/unknown reference            → snr_ft8eq = NULL
pwr_offset: tx_dbm known (WSPR) → (50 − tx_dbm)        (normalize to 100 W = 50 dBm)
            tx_dbm unknown      → 0                     (assume ~100 W; residual bias, §5)
```

### 4.5 Sampling

- **Training set**: all positives; negatives downsampled uniformly at random
  to **3:1 negative:positive per (band, UTC date) stratum** (keep all if
  fewer). Every row carries `sample_weight = 1/sampling_rate` (positives:
  1.0) so unbiased rates are recoverable. RNG: PCG64 seeded with
  `sha256(f"{band}|{date}") & 0xFFFFFFFF` — deterministic across
  implementations.
- **Evaluation**: the FULL labeled set, never sampled.

**Storage** (`lake/labels/band=…/date=…/*.parquet`): `window_start, tx_field,
rx_field, band, open (int8), n_spots (int32), n_monitors, n_tx_stations,
evidence_tier, snr_ft8eq_p50 (float32, nullable), sample_weight (float32),
split_tag (utf8, nullable)`. (Horizon is a training-time join offset, not a
label property — labels are stored once, not per horizon.
**[deviation — simplification]** ARCHITECTURE's lake sketch shows
`labels/horizon=…`; that duplicates identical data six times.)

## 5. Known residual biases (documented, not solved)

1. **Antenna directionality**: a "monitoring" receiver with a beam pointed
   away yields false negatives. Unobservable in v1.
2. **Power heterogeneity**: tx evidence may come from a QRP station; the path
   might be workable at 100 W. Partially mitigated by the WSPR tier
   (calibrated power) — slice eval by `evidence_tier`.
3. **Behavioral targeting**: operators transmit toward regions they expect
   open (gray-line chasing, contest multipliers), inflating positive rates
   where climatology already predicts openings. Affects both our labels and
   any baseline scored on them equally.
4. **Intra-field geography**: fields are 20°×10°; "open to EM" may mean open
   to one corner of EM. Bounded by cell size; revisit at grid4 granularity.
5. **Mode decode-threshold asymmetry**: FT8 (−21 dB) proves marginal paths CW
   (~0 dB) cannot; openness is implicitly "open for the most sensitive active
   mode." Mode_class matching (§3, §4.1) prevents the cross-mode version of
   this bias but not the within-class FT8-vs-JT65 version.
6. **FT8 watering-hole**: single sub-band per band; frequency-dependent
   effects within a band are unobserved.
7. **Receiver noise-floor variance**: urban vs rural monitors differ by
   >20 dB; `n_monitors` correlates with effective sensitivity.

## 6. Leakage rules & QA

### Leakage (hard rules)

1. Autoregressive features: max lookback **≤ 48 h** (current max used: 24 h).
   Blocked CV gap = `max(48h, max feature lookback)`. Adding a feature with
   longer lookback MUST widen the gap in `eval/splits.py`.
2. Climatology baseline, data-density `confidence` tiers, per-mode SNR offset
   refinements, isotonic calibrators: fitted on train folds only.
3. Receiver-uptime tables are window-local (±30 min) label infrastructure —
   computed over the whole dataset, no fold restriction needed. The ±30 min
   padding crosses the fold gap by at most 30 min ≪ 48 h; acceptable.
4. Space weather features: as-of-available series only (estimated Kp, not
   definitive). Definitive series allowed for eval stratification only.
5. Storm stratification: a fold/day is **storm** iff any 3-h Kp ≥ 5. Headline
   eval MUST include ≥1 storm fold.

### QA sanity checks (run before any training; pipeline fails loudly)

| # | check | expectation |
|---|---|---|
| 1 | 20m diurnal, mid-lat paths 3–8 Mm | day/night open-rate ratio > 2 |
| 2 | 160m/80m diurnal, paths > 2 Mm | night/day ratio > 5 |
| 3 | 40m gray-line | open-rate local max within ±1 h of midpoint terminator vs midday, DX paths > 6 Mm |
| 4 | 6m sporadic-E | NH May–Jul open-rate ≥ 3× Nov–Jan (1–2.3 Mm paths) |
| 5 | Reciprocity | Pearson r of open-rate(TX→RX) vs (RX→TX) per (pair, band, month) > 0.6 |
| 6 | Solar cycle | monthly 10m DX open-rate vs F10.7 correlation > 0.5 (needs multi-year data) |
| 7 | Storm response | Kp ≥ 6: trans-polar paths (midpoint geomag lat > 60°) open-rate ≤ 50% of Kp ≤ 2 matched (band, hour, month) baseline |
| 8 | Volume/hygiene | per band/day: label counts within 5× of trailing 28-day median; RR73-grid rejects < 0.5% of spots; unlabeled-activity fraction reported and trended |

Checks 1–4 validate that labels encode ionospheric physics; 5 validates
symmetry; 6–7 validate space-weather response; 8 catches pipeline breakage.
Where multi-year history is unavailable (check 6), the check reports
"insufficient data" rather than passing silently.
