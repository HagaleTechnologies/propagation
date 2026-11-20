"""PRO-10 cqdx R2 archive parsing tests (AC1: row mapping + source filter)."""
import datetime as dt
import gzip
import json

import polars as pl
import pytest

from propagation.data.cqdx_r2 import (
    archive_rows_to_frame,
    iter_requested_days,
    parse_archive_object,
    parse_archive_payload,
)
from propagation.data.schema import SPOT_SCHEMA

_TS = 1780000000  # 2026-05-29T07:06:40Z
_BASE = {
    "timestamp": _TS,
    "frequency": 14074000,
    "band": "20m",
    "mode": "ft8",
    "dxCall": "k1bbb",
    "deCall": "w1aaa",
    "dxGrid": "dm79",
    "deGrid": "fn31",
    "snr": -10,
    "source": "pskreporter",
}


def _payload(**overrides):
    row = dict(_BASE)
    row.update(overrides)
    return row


def test_extract_cqdx_maps_archive_rows_and_filters_sources():
    rows = []
    cases = [
        (_payload(), True, None),
        # camelCase mirror of the same transmission reported via RBN: admitted,
        # still archived with cqdx provenance
        (_payload(source="rbn", frequency=14032000, mode="cw"), True, None),
        # manual logbook rows are not sensor-equivalent
        (_payload(source="user"), False, "non_automatic_source"),
        # DX cluster records' provenance is not an automatic report
        (_payload(source="cluster"), False, "non_automatic_source"),
        (_payload(source=None), False, "non_automatic_source"),
        # timestamp/frequency type errors
        (_payload(timestamp="not-a-number"), False, "invalid_timestamp"),
        (_payload(timestamp=None), False, "invalid_timestamp"),
        (_payload(frequency="abc"), True, None),  # freq is nullable in schema
        # RR73 artifact grid
        (_payload(dxGrid="RR73"), False, "rr73_grid"),
        # invalid coordinates (non-finite / out of range)
        (_payload(dxGrid=None, dxLat=999.0, dxLon=0.0), False, "no_usable_location"),
        (_payload(dxGrid=None, dxLat="NaN", dxLon=0.0), False, "no_usable_location"),
        # self spot
        (_payload(dxCall="W1AAA"), False, "self_spot"),
        # unsupported band
        (_payload(band="2m"), False, "unsupported_band"),
        # unknown source value among admitted shapes -> generic hygiene reject
        (_payload(mode="", snr="x"), False, "missing_required_field"),
    ]
    for payload, ok, reason in cases:
        if ok:
            row, r = parse_archive_payload(payload)
            assert r is None
            rows.append(row)
        else:
            row, r = parse_archive_payload(payload)
            assert row is None
            assert r == reason, payload

    # coordinate-only rows derive a 4-char Maidenhead grid
    row, r = parse_archive_payload(_payload(dxGrid=None, dxLat=40.5, dxLon=-74.5))
    assert r is None and row["dx_grid"] == "FN20"
    # a grid row derives lat/lon from the normalized grid
    row, _ = parse_archive_payload(_payload())
    assert row["dx_grid"] == "DM79" and row["dx_lat"] == 39.5 and row["dx_lon"] == -105.0
    # provenance: payload origin never overrides archive origin
    assert row["source"] == "cqdx"
    # nullable fields stay null
    assert row["tx_dbm"] is None and row["dx_field"] is None and row["de_field"] is None
    # snr type errors fall back to null, not rejection
    row, _ = parse_archive_payload(_payload(snr="bad"))
    assert row["snr_db"] is None

    # gzip NDJSON path: one object mixes qualifying rows, rejections, malformed JSON
    lines = [
        json.dumps(_payload(dxCall="K1BBB")),
        json.dumps(_payload(dxCall="K2CCC", source="user")),
        "{not json",
        json.dumps(_payload(dxCall="K3CCC")),
    ]
    body = gzip.compress(("\n".join(lines) + "\n").encode())
    result = parse_archive_object(body)
    assert result.n_qualifying == 2
    assert result.rejection_counts == {"non_automatic_source": 1, "malformed_json": 1}

    frame = archive_rows_to_frame(result.rows)
    assert frame.schema == pl.DataFrame(schema=SPOT_SCHEMA).schema
    assert frame["source"].unique().to_list() == ["cqdx"]
    assert frame["ts"].dtype == pl.Datetime("us", "UTC")
    assert frame["ts"][0] == dt.datetime.fromtimestamp(_TS, tz=dt.timezone.utc)
    assert archive_rows_to_frame([]).schema == pl.DataFrame(schema=SPOT_SCHEMA).schema


def test_parse_archive_object_truncated_gzip_raises():
    body = gzip.compress(json.dumps(_BASE).encode())
    with pytest.raises(Exception):
        parse_archive_object(body[:-4])


def test_iter_requested_days_half_open():
    days = list(iter_requested_days(dt.date(2026, 6, 30), dt.date(2026, 7, 2)))
    assert days == [dt.date(2026, 6, 30), dt.date(2026, 7, 1)]
    assert list(iter_requested_days(dt.date(2026, 6, 1), dt.date(2026, 6, 1))) == []


def test_gzip_object_streaming_decode_matches_batch():
    # one line at a time vs whole-object decode must agree on row count
    lines = [json.dumps(_payload(dxCall=f"K{i}BBB")) for i in range(50)]
    result = parse_archive_object(gzip.compress(("\n".join(lines) + "\n").encode()))
    assert result.n_qualifying == 50
