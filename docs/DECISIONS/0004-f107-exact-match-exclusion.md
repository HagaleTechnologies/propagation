# 0004: F10.7's as-of join excludes exact-timestamp matches; other space-weather fields don't

## Status

Accepted.

## Context

`propagation.features.spaceweather.add_spaceweather_features` (M2 Task 4)
joins OMNI2 hourly space-weather data onto each label row via
`join_asof(strategy="backward", tolerance="1h")`. The brief's original
reference code used one shared join (`other_now`, tie-inclusive by default:
`allow_exact_matches=True`) for `kp_now`, `bz_gsm_now`, `solar_wind_speed_now`,
`dst_now`, and `f107_daily` alike.

That produced a contradiction the task's own two tests exposed: a test
expecting `kp_now` to equal the value at an OMNI row exactly matching
`window_start` (tie-inclusive) passed, while a test expecting
`f107_daily` to equal the *previous* day's smoothed value at the same kind of
exact-match boundary failed — polars' `join_asof(allow_exact_matches=True)`
correctly returns the exact-match row's value for both, so both tests can't
pass under one shared join with one `allow_exact_matches` setting.

## Decision

Split `f107` into its own `join_asof` call with `allow_exact_matches=False`;
`kp_now`/`bz_gsm_now`/`solar_wind_speed_now`/`dst_now` keep the tie-inclusive
default. Physical justification: Kp, Bz, solar wind speed, and Dst are
near-real-time telemetry — an OMNI2 row timestamped exactly at
`window_start` reflects a value genuinely known at that instant. F10.7, by
contrast, is a once-daily ground measurement (observed near local noon) that
OMNI2 backfills across all 24 hourly rows of its UTC day; at the exact top of
a UTC day boundary, that day's own F10.7 reading has not actually been taken
yet, so treating it as already-known at hour 0 would leak information
forward by up to ~24h in the worst case.

## Consequences

- `f107_daily`/`f107_smoothed_27d` are the only two `FEATURE_COLUMNS`
  produced by an as-of join with `allow_exact_matches=False`; every other
  as-of-joined space-weather column uses the tie-inclusive default. This
  asymmetry is intentional, not an oversight — see
  `src/propagation/features/spaceweather.py`'s docstring and Task 4's test
  suite (`tests/features/test_spaceweather.py::test_f107_smoothed_27d_is_a_trailing_not_centered_mean`)
  for the concrete boundary case this covers.
- If a future space-weather field with once-daily (or coarser) cadence is
  added, apply the same `allow_exact_matches=False` treatment rather than
  reusing the near-real-time fields' join — the cadence of the underlying
  measurement, not convenience, decides which join semantics apply.
