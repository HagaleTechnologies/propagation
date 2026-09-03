import polars as pl

from propagation.data.schema import SPOT_SCHEMA, SUPPORTED_BANDS, normalize_spot_columns


def test_supported_bands_matches_spec():
    assert SUPPORTED_BANDS == {
        "160m", "80m", "60m", "40m", "30m", "20m", "17m", "15m", "12m", "10m", "6m",
    }


def test_spot_schema_has_required_columns():
    # dx_field/rx_field naming resolved during implementation; both tx and rx
    # fields for each side must be present under *some* consistent names.
    assert {"source", "ts", "band", "mode", "dx_call", "de_call", "snr_db"} <= set(
        SPOT_SCHEMA
    )


def test_normalize_spot_columns_adds_missing_columns_as_null():
    df = pl.DataFrame({"source": ["wsprnet"], "band": ["20m"]})
    normalized = normalize_spot_columns(df)
    assert set(normalized.columns) == set(SPOT_SCHEMA)
    assert normalized["dx_grid"].to_list() == [None]


def test_normalize_spot_columns_always_produces_schema_order():
    """Two frames built independently (one with every column present from
    the start, one padded with missing columns appended at the end) must
    come out in the SAME column order -- this is what makes
    pl.concat(..., how="vertical_relaxed") safe between them.
    vertical_relaxed matches columns positionally, not by name, so an order
    mismatch between an in-memory frame and one read back from disk crashed
    PRO-9's live accumulator (see pskreporter.py's write_hourly_parquet)."""
    already_complete = pl.DataFrame(schema=SPOT_SCHEMA)
    missing_a_few = pl.DataFrame({"source": ["rbn"], "band": ["20m"], "dx_call": ["W1AW"]})
    assert normalize_spot_columns(already_complete).columns == list(SPOT_SCHEMA)
    assert normalize_spot_columns(missing_a_few).columns == list(SPOT_SCHEMA)


def test_normalize_spot_columns_is_a_noop_on_an_already_normalized_frame():
    df = pl.DataFrame(schema=SPOT_SCHEMA)
    assert normalize_spot_columns(df).columns == list(SPOT_SCHEMA)
