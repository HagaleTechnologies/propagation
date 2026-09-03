import zipfile
from pathlib import Path

import pytest

from propagation.data.rbn import (
    extract_rbn,
    parse_rbn_row,
    strip_skimmer_suffix,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "rbn_sample.csv"

# Fake resolver standing in for dxentity.DXCC -- distinct, well-separated
# coordinates per (post-suffix-strip) callsign so distance_too_short never
# fires spuriously in these tests. Real-world resolution is country-centroid
# (see rbn.py's module docstring); this fixture uses station-level spread to
# isolate extract_rbn's own parsing/filtering/dedup behavior from that
# separate, already-documented precision gap.
_FAKE_LOCATIONS = {
    "BD8CS": (39.9, 116.4),
    "BY8DX": (31.2, 121.5),
    "W1NT": (41.5, -71.3),
    "K9EI": (41.6, -87.0),
    "TI7W": (10.0, -84.0),
    "ND7K": (36.0, -112.0),
    "N4KS": (33.0, -84.0),
    "KM3T": (39.0, -76.0),
}


def _fake_resolver(call: str):
    return _FAKE_LOCATIONS.get(call)


def test_strip_skimmer_suffix():
    assert strip_skimmer_suffix("KM3T-3") == "KM3T"
    assert strip_skimmer_suffix("W1NT-2") == "W1NT"
    assert strip_skimmer_suffix("TI7W") == "TI7W"


def test_parse_rbn_row_maps_fields():
    row = {
        "callsign": "W1NT-2", "de_pfx": "K", "de_cont": "NA", "freq": "14031.4",
        "band": "20m", "dx": "K9EI", "dx_pfx": "K", "dx_cont": "NA", "mode": "CQ",
        "db": "33", "date": "2024-01-15 00:00:08", "speed": "20", "tx_mode": "CW",
    }
    parsed = parse_rbn_row(row, _fake_resolver)
    assert parsed["source"] == "rbn"
    assert parsed["mode"] == "CW"
    assert parsed["band"] == "20m"
    assert parsed["dx_call"] == "K9EI"
    assert parsed["de_call"] == "W1NT"  # skimmer suffix stripped
    assert parsed["dx_grid"] is None and parsed["de_grid"] is None
    assert parsed["dx_lat"] == 41.6 and parsed["dx_lon"] == -87.0
    assert parsed["de_lat"] == 41.5 and parsed["de_lon"] == -71.3
    assert parsed["snr_db"] == 33
    assert parsed["freq_hz"] == 14031400
    assert parsed["tx_dbm"] is None


def test_parse_rbn_row_rejects_missing_fields():
    assert parse_rbn_row({"callsign": "W1NT"}, _fake_resolver) is None


def test_parse_rbn_row_unresolved_location_is_null():
    row = {
        "callsign": "ZZ1ZZZ", "de_pfx": "?", "de_cont": "??", "freq": "14025",
        "band": "20m", "dx": "K9EI", "dx_pfx": "K", "dx_cont": "NA", "mode": "CQ",
        "db": "10", "date": "2024-01-15 00:00:00", "speed": "20", "tx_mode": "CW",
    }
    parsed = parse_rbn_row(row, _fake_resolver)
    assert parsed["de_lat"] is None and parsed["de_lon"] is None


@pytest.fixture
def zip_fixture(tmp_path):
    zip_path = tmp_path / "20240115.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("20240115.csv", FIXTURE.read_text())
    return zip_path


def test_extract_rbn_filters_band_and_hygiene(zip_fixture):
    result = extract_rbn(zip_fixture, band="20m", resolve_location=_fake_resolver)
    assert result.n_lines_read == 7  # 6 real rows + 1 malformed row
    assert result.n_parsed == 5  # 20m rows: BD8CS, W1NT-2, TI7W, KM3T-3, K9EI-self
    assert result.rejection_counts.get("self_spot") == 1
    assert result.n_qualifying == 4
    assert set(result.spots["de_call"].to_list()) == {"BD8CS", "W1NT", "TI7W", "KM3T"}


def test_extract_rbn_excludes_other_bands(zip_fixture):
    result = extract_rbn(zip_fixture, band="40m", resolve_location=_fake_resolver)
    assert result.n_parsed == 1
    assert result.spots["dx_call"][0] == "N4KS"


def test_extract_rbn_missing_csv_member_raises(tmp_path):
    empty_zip = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w"):
        pass
    with pytest.raises(ValueError, match="no .csv member"):
        extract_rbn(empty_zip, band="20m", resolve_location=_fake_resolver)


def test_extract_rbn_column_order_matches_schema_regardless_of_qualifying_rows(zip_fixture):
    """A source with qualifying rows and one with none must produce the SAME
    column order (SPOT_SCHEMA's) -- otherwise concatenating this extractor's
    output with another source's (e.g. eval_m3.py --include-rbn merging RBN
    alongside WSPRnet, PRO-8) can crash pl.concat(..., how="vertical_relaxed")
    exactly as it did for pskreporter.py's write_hourly_parquet during PRO-9's
    live soak test."""
    from propagation.data.schema import SPOT_SCHEMA

    has_rows = extract_rbn(zip_fixture, band="20m", resolve_location=_fake_resolver)
    no_rows = extract_rbn(zip_fixture, band="99m", resolve_location=_fake_resolver)
    assert has_rows.n_qualifying > 0
    assert no_rows.n_qualifying == 0
    assert has_rows.spots.columns == list(SPOT_SCHEMA)
    assert no_rows.spots.columns == list(SPOT_SCHEMA)
