import polars as pl

from propagation.data.hygiene import mode_class_for


def _with_fields(spots: pl.DataFrame) -> pl.DataFrame:
    return spots.with_columns(
        pl.col("ts").dt.truncate("15m").alias("window_start"),
        pl.col("dx_grid").str.slice(0, 2).alias("dx_field"),
        pl.col("de_grid").str.slice(0, 2).alias("de_field"),
        pl.col("mode").map_elements(mode_class_for, return_dtype=pl.Utf8).alias("mode_class"),
    )


def build_transmit_evidence(spots: pl.DataFrame) -> pl.DataFrame:
    """docs/SPEC-labeling.md sec 4.1. No padding — exact window only."""
    working = _with_fields(spots).filter(pl.col("mode_class") != "other")
    return (
        working.group_by(["window_start", "dx_field", "band", "mode_class"])
        .agg(
            pl.len().alias("n_evidence_reports"),
            (pl.col("source") == "wsprnet").any().alias("_has_wspr"),
        )
        .rename({"dx_field": "tx_field"})
        .with_columns(
            pl.when(pl.col("_has_wspr")).then(pl.lit("wspr")).otherwise(pl.lit("spot")).alias(
                "evidence_tier"
            )
        )
        .drop("_has_wspr")
    )


def build_universe(spots: pl.DataFrame, uptime: pl.DataFrame) -> pl.DataFrame:
    """docs/SPEC-labeling.md sec 2, sec 4.3. Universe = positive OR N-eligible cells."""
    fielded = _with_fields(spots)

    positives = (
        fielded.group_by(["window_start", "dx_field", "de_field", "band"])
        .agg(pl.len().alias("n_spots"))
        .rename({"dx_field": "tx_field", "de_field": "rx_field"})
    )

    tx_evidence = build_transmit_evidence(spots)
    monitors_by_rx = uptime.rename({"de_field": "rx_field"}).group_by(
        ["window_start", "rx_field", "band", "mode_class"]
    ).agg(pl.col("de_call").n_unique().alias("n_monitors"))

    n_eligible_pairs = monitors_by_rx.join(
        tx_evidence, on=["window_start", "band", "mode_class"], how="inner"
    ).group_by(["window_start", "tx_field", "rx_field", "band"]).agg(
        pl.col("n_monitors").sum().alias("n_monitors"),
        pl.col("n_evidence_reports").sum().alias("n_tx_stations"),
        (pl.col("evidence_tier") == "wspr").any().alias("_has_wspr"),
    ).with_columns(
        pl.lit(True).alias("is_n_eligible"),
        pl.when(pl.col("_has_wspr")).then(pl.lit("wspr")).otherwise(pl.lit("spot")).alias(
            "evidence_tier"
        ),
    ).drop("_has_wspr")

    universe = positives.join(
        n_eligible_pairs,
        on=["window_start", "tx_field", "rx_field", "band"],
        how="full",
        coalesce=True,
    ).with_columns(
        pl.col("n_spots").fill_null(0),
        pl.col("is_n_eligible").fill_null(False),
        pl.col("n_monitors").fill_null(0),
        pl.col("n_tx_stations").fill_null(0),
        pl.col("evidence_tier").fill_null("spot"),
    ).with_columns((pl.col("n_spots") > 0).alias("is_positive"))

    return universe.filter(pl.col("is_positive") | pl.col("is_n_eligible")).select(
        "window_start", "tx_field", "rx_field", "band", "is_positive", "is_n_eligible",
        "n_spots", "n_monitors", "n_tx_stations", "evidence_tier",
    )


def unlabeled_activity_fraction(spots: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    """Engineering proxy (SPEC/ROADMAP require reporting this but give no closed-form
    formula): among (window, field) with any qualifying-spot activity (as tx or rx),
    form the candidate active_field x active_field universe per (window, band); the
    unlabeled fraction is 1 - |actual universe| / |candidate universe|, aggregated to
    band/date. This measures how much of the activity-adjacent space we could not
    resolve into a positive or N-eligible label (missing monitor or tx evidence on the
    other side)."""
    fielded = _with_fields(spots)
    tx_active = fielded.select(["window_start", "band", pl.col("dx_field").alias("field")])
    rx_active = fielded.select(["window_start", "band", pl.col("de_field").alias("field")])
    active_fields = pl.concat([tx_active, rx_active]).unique()

    candidates = active_fields.join(active_fields, on=["window_start", "band"], how="inner")
    candidates = candidates.rename({"field": "tx_field", "field_right": "rx_field"}).unique()

    candidates = candidates.with_columns(pl.col("window_start").dt.date().cast(pl.Utf8).alias("date"))
    universe_dated = universe.with_columns(
        pl.col("window_start").dt.date().cast(pl.Utf8).alias("date")
    )

    candidate_counts = candidates.group_by(["band", "date"]).agg(pl.len().alias("n_candidates"))
    universe_counts = universe_dated.group_by(["band", "date"]).agg(pl.len().alias("n_universe"))

    report = candidate_counts.join(universe_counts, on=["band", "date"], how="left").with_columns(
        pl.col("n_universe").fill_null(0)
    )
    return report.with_columns(
        (1.0 - pl.col("n_universe") / pl.col("n_candidates")).alias("unlabeled_fraction")
    ).select("band", "date", "unlabeled_fraction")
