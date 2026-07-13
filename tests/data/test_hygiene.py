import datetime as dt


from propagation.data.hygiene import (
    base_call,
    has_excluded_suffix,
    is_qualifying_spot,
    is_valid_callsign,
    mode_class_for,
    normalize_grid,
)


def test_mode_class_digi():
    assert mode_class_for("FT8") == "digi"
    assert mode_class_for("wspr") == "digi"


def test_mode_class_cw():
    assert mode_class_for("CW") == "cw"
    assert mode_class_for("RTTY") == "cw"


def test_mode_class_other():
    assert mode_class_for("SSB") == "other"


def test_normalize_grid_truncates_grid6():
    assert normalize_grid("EM12ab") == "EM12"


def test_normalize_grid_accepts_field_only():
    assert normalize_grid("EM") == "EM"


def test_normalize_grid_rejects_rr73():
    assert normalize_grid("RR73") is None


def test_normalize_grid_rejects_garbage():
    assert normalize_grid("9999") is None
    assert normalize_grid("") is None
    assert normalize_grid(None) is None


def test_is_valid_callsign_accepts_normal():
    assert is_valid_callsign("K1JT")
    assert is_valid_callsign("W6SZ")
    assert is_valid_callsign("2E0DLC")


def test_is_valid_callsign_accepts_suffixed():
    assert is_valid_callsign("K1JT/P")
    assert is_valid_callsign("K1JT/QRP")


def test_is_valid_callsign_rejects_garbage():
    assert not is_valid_callsign("!!!")
    assert not is_valid_callsign("RR73")


def test_is_valid_callsign_strips_hash_markers():
    assert is_valid_callsign("<K1JT>")


def test_has_excluded_suffix():
    assert has_excluded_suffix("KL7XYZ/MM")
    assert has_excluded_suffix("N0CALL/AM")
    assert not has_excluded_suffix("K1JT/P")


def test_base_call_strips_any_suffix():
    assert base_call("K1JT/P") == "K1JT"
    assert base_call("K1JT/QRP") == "K1JT"
    assert base_call("K1JT") == "K1JT"


def _row(**overrides):
    row = {
        "ts": dt.datetime(2026, 6, 1, 12, 0, tzinfo=dt.timezone.utc),
        "band": "20m",
        "mode": "WSPR",
        "dx_call": "K1JT",
        "de_call": "W6SZ",
        "dx_grid": "FN20",
        "de_grid": "DM14ed",
        "dx_lat": None,
        "dx_lon": None,
        "de_lat": None,
        "de_lon": None,
    }
    row.update(overrides)
    return row


def test_is_qualifying_spot_accepts_valid_row():
    ok, reason = is_qualifying_spot(_row())
    assert ok
    assert reason is None


def test_is_qualifying_spot_rejects_unsupported_band():
    ok, reason = is_qualifying_spot(_row(band="2m"))
    assert not ok
    assert reason == "unsupported_band"


def test_is_qualifying_spot_rejects_invalid_callsign():
    ok, reason = is_qualifying_spot(_row(dx_call="!!!"))
    assert not ok
    assert reason == "invalid_callsign"


def test_is_qualifying_spot_rejects_mm_suffix():
    ok, reason = is_qualifying_spot(_row(dx_call="KL7XYZ/MM"))
    assert not ok
    assert reason == "mm_am_suffix"


def test_is_qualifying_spot_rejects_self_spot():
    ok, reason = is_qualifying_spot(_row(dx_call="K1JT/P", de_call="K1JT"))
    assert not ok
    assert reason == "self_spot"


def test_is_qualifying_spot_rejects_rr73_grid():
    ok, reason = is_qualifying_spot(_row(dx_grid="RR73"))
    assert not ok
    assert reason == "rr73_grid"


def test_is_qualifying_spot_rejects_no_location():
    ok, reason = is_qualifying_spot(_row(dx_grid=None, dx_lat=None, dx_lon=None))
    assert not ok
    assert reason == "no_usable_location"


def test_is_qualifying_spot_accepts_latlon_fallback():
    ok, reason = is_qualifying_spot(
        _row(dx_grid=None, dx_lat=42.0, dx_lon=-71.0)
    )
    assert ok, reason


def test_is_qualifying_spot_rejects_too_close():
    # Same field, ~0km apart
    ok, reason = is_qualifying_spot(_row(dx_grid="FN20", de_grid="FN20"))
    assert not ok
    assert reason == "distance_too_short"
