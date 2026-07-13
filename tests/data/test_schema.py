from propagation.data.schema import SUPPORTED_BANDS, SPOT_SCHEMA


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
