# cqdx R2 backfill (optional private accelerant)

The cqdx R2 archive is an **optional accelerant, never a dependency**: the
public-only pipeline (WSPRnet + RBN + PSKReporter) must keep clearing M2's
bar with or without it. Everything in this document is opt-in and installs
nothing by default.

## What the archive holds

cqdx's private Cloudflare R2 bucket `cqdx-archive` stores mirrored spot
reports under `raw/YYYY-MM-DD/HH/{ulid}.ndjson.gz` — one gzipped NDJSON
object per hour, each line an enriched camelCase spot record
(`timestamp`, `frequency`, `band`, `mode`, `dxCall`, `dxGrid`,
`dxLat`/`dxLon`, `deCall`, `deGrid`, `deLat`/`deLon`, `snr`, `source`, …).
This layout and schema are the merged dispensa Q-0062 answer (dispensa PRs
#129/#130). The same bucket also contains PII-bearing `pending-imports/`
logbooks — access must be scoped to `raw/` only.

`scripts/extract_cqdx.py` admits only confirmed automatic reports (payload
`source` of `pskreporter` or `rbn`); manual `source:'user'` rows and DX
cluster records are filtered out. Surviving rows are written with archive
provenance `source=cqdx`, the lowest dedup priority in
`propagation.data.dedup.SOURCE_PRIORITY`.

## Backfilling

Prerequisites (owner-side, see dispensa Q-0062):

1. A read-only temporary S3 credential scoped to `raw/` (Cloudflare R2
   temporary credentials with a prefix condition; the owner attests the
   server-side scope — a client-side prefix check cannot confer it).
2. The R2 account ID (32 hex characters).

Then:

```console
$ export AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… AWS_SESSION_TOKEN=…
$ uv run --extra cqdx-r2 python scripts/extract_cqdx.py \
    --account-id 0123456789abcdef0123456789abcdef \
    --start-date 2026-06-01 --end-date 2026-06-02 --out-dir data/cqdx-june
Wrote 120 cqdx spots to data/cqdx-june (1 day, 3 objects, 8 rejected).
```

- `--start-date` is inclusive, `--end-date` exclusive, in UTC archive
  partition dates. Spot observation time (not archive date) later selects
  the evaluation month.
- boto3 lives behind the optional `cqdx-r2` extra; the default dependency
  set and every public evaluation run without it.
- Missing credentials exit 2 before contacting R2; an archive read failure
  exits 1 and publishes nothing (staging is removed); an empty range exits
  1. The output directory must be fresh — an existing snapshot is never
  overwritten.
- Output: one `cqdx-YYYY-MM-DD.parquet` per non-empty archive day, exact
  `SPOT_SCHEMA` types, deduplicated within each archive day.
- Keep the destination under gitignored `data/` or on operator-controlled
  private storage; never redistribute raw rows.

## The acceptance comparison

`scripts/eval_m3.py`'s acceptance mode scores the public pipeline and the
R2-augmented pipeline on the SAME public evaluation cohort, so an apparent
gain cannot come from changing the evaluation population.

Public-only (the milestone requirement; no R2 access or SDK needed):

```console
$ uv run python scripts/eval_m3.py --bands 20m --horizons 0 \
    --train-months 2026-06 --eval-months 2026-08 2026-09 2026-10 \
    --data-dir data --p533-workers 3 --acceptance-pskreporter-dir data/raw/pskreporter
PASS public M2 gate on 20m at h=0. Report: data/reports/m3/acceptance/20m/public.json
```

With the optional private arm added to training:

```console
$ uv run python scripts/eval_m3.py --bands 20m --horizons 0 \
    --train-months 2026-06 --eval-months 2026-08 2026-09 2026-10 \
    --data-dir data --p533-workers 3 --acceptance-pskreporter-dir data/raw/pskreporter \
    --cqdx-dir data/cqdx-june
PASS public M2 gate and cqdx Brier comparison on 20m at h=0. Report: data/reports/m3/acceptance/20m/augmented.json
```

Requirements and behavior:

- One band from `{20m, 10m}`, horizon 0, at least three distinct held-out
  months, non-overlapping train/eval months, a ≥48h blocked-CV gap on the
  actual label timestamps (enforced on both arms), and at least one
  evaluation row inside a definitive Kp ≥ 5 block. Violations exit 2 before
  any scoring.
- Acceptance mode always includes RBN. WSPRnet archives download as in the
  default mode; PSKReporter is read from the public accumulator's hourly
  `pskreporter-YYYY-MM-DDTHH.parquet` files
  (`scripts/run_pskreporter_accumulator.py`).
- The public gate requires the public GBT to score strictly below BOTH
  climatology and P.533 on BOTH Brier and log-loss over the full unsampled
  public label set. A valid comparison that misses a threshold exits 1.
- The cqdx comparison adds only TRAINING rows that survive cross-source
  deduplication (public priority wins ties; private holdout-month rows are
  never loaded). It compares the full-precision paired Brier difference
  `Brier(augmented) − Brier(public)` on identical keys, labels and
  features, and passes at zero or below. Missing evidence (absent sources,
  empty months, model abstentions, no surviving private rows) is `unverified`
  and exits 2.
- Reports are written atomically per band and mode
  (`public.json` / `augmented.json`), including a SHA-256 digest of the
  evaluation cohort's keys and labels for comparing runs.

## Live-run status

No R2 access has been provisioned yet (owner action per PRO-10). The
implementation gate covers ingest and scoring with fixtures; the live
benchmark procedure — inventory matching public archives, freeze splits per
band, run public-only first, then the augmented run on the same frozen
inputs — is recorded in the PRO-10 plan and must run before any milestone
claim about the private arm.
