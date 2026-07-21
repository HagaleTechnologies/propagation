import gzip
from pathlib import Path

import pytest

from propagation.data.wsprnet import (
    WSPR_BAND_CODE_TO_BAND,
    extract_wsprnet,
    parse_wsprnet_row,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "wspr_sample.csv"


def test_band_code_mapping():
    assert WSPR_BAND_CODE_TO_BAND[14] == "20m"
    assert WSPR_BAND_CODE_TO_BAND[10] == "30m"
    assert WSPR_BAND_CODE_TO_BAND[50] == "6m"


def test_parse_wsprnet_row_maps_fields():
    line = "1012028,1717243320,K1JT,FN20,-20,14.097100,W6SZ,DM14ed,20,0,3086,79,14,0.7_r752,0"
    row = parse_wsprnet_row(line)
    assert row["source"] == "wsprnet"
    assert row["mode"] == "WSPR"
    assert row["band"] == "20m"
    assert row["dx_call"] == "W6SZ"
    assert row["de_call"] == "K1JT"
    assert row["dx_grid"] == "DM14ED"
    assert row["de_grid"] == "FN20"
    assert row["snr_db"] == -20
    assert row["tx_dbm"] == 20
    assert row["freq_hz"] == 14097100


def test_parse_wsprnet_row_rejects_unsupported_band():
    line = "1,1717243320,A,FN20,-20,144.174,B,DM14,37,0,10,10,144,ver,0"
    assert parse_wsprnet_row(line) is None


def test_parse_wsprnet_row_rejects_malformed():
    assert parse_wsprnet_row("garbage,not,enough,fields") is None


@pytest.fixture
def gz_fixture(tmp_path):
    gz_path = tmp_path / "wsprspots-2024-06.csv.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write(FIXTURE.read_text())
    return gz_path


def test_extract_wsprnet_filters_band_and_hygiene(gz_fixture):
    result = extract_wsprnet(gz_fixture, band="20m")
    assert result.n_lines_read == 4
    assert result.n_parsed == 3  # 3 lines are band=20m (rows 1, 2, 4)
    assert result.n_qualifying == 1  # rows 1+2 dedup to 1; row 4 rejected
    assert result.rejection_counts.get("invalid_callsign") == 1
    assert result.spots.height == 1
    assert result.spots["dx_call"][0] == "W6SZ"
    assert result.spots["snr_db"][0] == -9  # higher-snr decode wins dedup


def test_extract_wsprnet_empty_result_has_correct_schema(tmp_path):
    gz_path = tmp_path / "empty.csv.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write("")
    result = extract_wsprnet(gz_path, band="20m")
    assert result.spots.height == 0
    assert result.n_lines_read == 0


def test_extract_wsprnet_chunking_matches_single_chunk_result(gz_fixture):
    # chunk_size=1 forces every qualifying row into its own flush -- the
    # concatenated, deduped result must be identical to one giant chunk
    # (the pre-fix behavior), proving the memory-bounding change doesn't
    # alter what gets extracted.
    chunked = extract_wsprnet(gz_fixture, band="20m", chunk_size=1)
    unchunked = extract_wsprnet(gz_fixture, band="20m", chunk_size=1_000_000)
    assert chunked.spots.equals(unchunked.spots)
    assert chunked.n_lines_read == unchunked.n_lines_read
    assert chunked.n_parsed == unchunked.n_parsed
    assert chunked.rejection_counts == unchunked.rejection_counts
