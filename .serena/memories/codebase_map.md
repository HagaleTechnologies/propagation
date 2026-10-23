# propagation codebase map

Starter orientation memory (PRO-13). Canon lives in CLAUDE.md's "Documents (read in
this order)"; this file only maps where code lives. Replace it with Serena's own
onboarding output when a session with the Serena MCP server runs onboarding.

## Package: src/propagation/

- data/ — extractors and spot hygiene. wsprnet.py (`extract_wsprnet`), rbn.py
  (`extract_rbn`), pskreporter.py (`PSKReporterAccumulator`, live MQTT),
  spaceweather.py (OMNI2 Kp, F10.7, Bz, Dst fetch). dedup.py (`dedup_spots`),
  hygiene.py (`mode_class_for`, `is_qualifying_spot`), geo.py (Maidenhead grid
  <-> lat/lon), schema.py (column normalization). lake.py is the generic
  hive-partitioned Parquet writer (`write_partitioned`, `register_view`);
  callers choose the band=/date= keys.
- features/ — monitor-normalized labeling lives in universe.py (`build_universe`)
  plus uptime.py (`build_receiver_uptime`); labels.py (`build_labels`); sampling.py
  (`sample_labels`) is the training-only band/date-stratified negative subsampler
  with inverse-rate `sample_weight`. matrix.py (`build_feature_matrix`) assembles
  solar.py, spaceweather.py, geometry.py and history.py features.
- models/ — climatology.py (`ClimatologyModel`), gbt.py (`GBTModel`: LightGBM +
  isotonic calibration), p533.py (`P533Model`, wraps the vendored ITURHFProp
  binary), p533_build.py (the `build-p533` console script).
- eval/ — splits.py (`blocked_time_series_folds`, >=48h gap), metrics.py (Brier,
  log-loss, reliability, PR-AUC), stratify.py (storm/quiet windows from Kp),
  report.py (`write_headline_report`).
- qa/ — checks.py (`run_qa_checks`: physics-grounded QA gates).

## Elsewhere

- scripts/ — entry points (run_m0.py, eval_m1.py..eval_m3.py,
  run_pskreporter_accumulator.py and others); operational tooling, not library code.
- tests/ — mirrors the package layout; fixtures in tests/fixtures/.
- contracts/ — prediction-surface.v1 JSON schema (the only cqdx integration surface).
- baselines/p533/upstream/ — vendored ITU ITURHFProp C/Fortran. Never edited;
  excluded from the Serena index and from ruff.
- docs/DECISIONS/ — normative ADRs. wiki/ — descriptive notes (wiki/INDEX.md).
