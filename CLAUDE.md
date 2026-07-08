# propagation

ML-based HF propagation nowcasting: train on real path reports (WSPRnet, RBN,
PSKReporter) to predict band openings, benchmarked against ITU-R P.533/VOACAP —
the 1980s empirical models the hobby still runs on. The novel methodological
contribution (and publication target) is the monitor-normalized negative
sampling that handles observation bias.

## Status

Design phase complete; no implementation yet. Next step is M0 in ROADMAP.md
(one month of one band extracted to Parquet + climatology baseline evaluated).

## Documents (read in this order)

- `README.md` — thesis, licensing & boundaries
- `ARCHITECTURE.md` — task formulation, data pipeline, feature engineering,
  modeling ladder, evaluation design
- `docs/SPEC-labeling.md` — the precise labeling methodology: universe
  definition, positive/negative rules, receiver-uptime tables, mode-class
  matching, leakage rules, physics-grounded QA gates. Two independent
  implementations must produce identical label sets from this spec.
- `contracts/prediction-surface.v1.schema.json` + `docs/SPEC-contract-notes.md`
  — the versioned public prediction contract (columnar, quantized encoding)
- `ROADMAP.md` — milestones M0–M4 with acceptance criteria

## Key constraints

- **Open/closed boundary (deliberate):** this repo will likely be open source
  (MIT/Apache-2.0 dual); cqdx will likely remain closed. Everything here must
  be reproducible from PUBLIC data (WSPRnet archives, RBN CSVs, PSKReporter
  MQTT, NOAA SWPC). The private cqdx R2 spot archive is an optional extractor
  behind a pluggable interface — an accelerant, never a dependency. No cqdx
  code imports, ever.
- The P.533 baseline vendors its own ITURHFProp build so the headline
  comparison is reproducible without cqdx access.
- cqdx integration = the JSON contract only, consumed by a thin closed adapter
  on the cqdx side (which may reuse cqdx's existing propagation-grid plumbing).
  Contract belongs in `dispensa` too (ADR pending).
- Stack: Python 3.11+/uv, DuckDB + Parquet (partitioned band=/date=), LightGBM
  first; spatiotemporal models gated on LightGBM beating P.533.
- Evaluation: blocked time-series CV with ≥48h gap; eval always on the full
  unsampled label set. Minimum credible result: beat P.533 + climatology on
  Brier at h=0/+3h on 20/15/10m over ≥3 months including one geomagnetic storm.
- Data licensing: no raw spot redistribution; pipelines pull canonical archives.

## Knowledge wiki

`wiki/INDEX.md` is the map of accumulated knowledge — read it before deep
exploration; open pages relevant to your task. After substantive work, run
/wiki-update: distill new gotchas/decisions/corrections into the wiki (or
into docs/ if normative — the wiki points, it never restates). The wiki is
descriptive and always loses conflicts with code and docs/.

## Multi-agent hygiene

You are never alone in this repo — other agents may be working concurrently
in other clones, branches, or worktrees.

- **Start fresh:** `git fetch` and rebase onto `origin/main` before reading
  code or making decisions; stale context produces wrong work.
- **Claim before work:** search open PRs/issues first; open a draft PR early —
  the draft PR *is* the claim. Don't duplicate in-flight work.
- **Isolate:** always a branch (worktree preferred), never a shared checkout's
  main. Use per-session scratch dirs; don't bind fixed ports.
- **Flush at the end:** push (`--force-with-lease` only) and open/update your
  PR before finishing. Unpushed work is invisible work.
- **Main moves only by PR merge.**
