import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from propagation.data.schema import SUPPORTED_BANDS
from propagation.models import p533
from propagation.models.p533 import BAND_FREQ_MHZ, P533Model, P533Result, parse_report, render_input_card
from propagation.models.p533_build import binary_path


def test_every_supported_band_has_a_frequency():
    assert set(BAND_FREQ_MHZ) == SUPPORTED_BANDS


def test_frequencies_are_the_primary_ft8_dial_frequency():
    expected = {
        "160m": 1.840, "80m": 3.573, "60m": 5.357, "40m": 7.074,
        "30m": 10.136, "20m": 14.074, "17m": 18.100, "15m": 21.074,
        "12m": 24.915, "10m": 28.074, "6m": 50.313,
    }
    assert BAND_FREQ_MHZ == expected


FIXTURE = Path(__file__).parent.parent / "fixtures" / "p533_report_sample.txt"


def test_render_input_card_contains_all_parameters():
    card = render_input_card(
        tx_lat=35.0, tx_lon=-90.0, rx_lat=45.0, rx_lon=135.0,
        month=7, hour_utc=14, ssn=123.4, freq_mhz=14.074,
        data_dir=Path("/repo/baselines/p533/upstream/ITURHFProp/Data"),
    )
    for token in (
        "Path.L_tx.lat 35.0000", "Path.L_tx.lng -90.0000",
        "Path.L_rx.lat 45.0000", "Path.L_rx.lng 135.0000",
        "Path.month 7", "Path.SSN 123", "Path.frequency 14.0740",
        "Path.BW 2500.0", "Path.SNRr -21.0",
        'RptFilFormat "RPT_BCR | RPT_SNR"',
    ):
        assert token in card, token


def test_render_input_card_hour_convention():
    # ITURHFProp hours are 1..24 UTC (verify against vendored README; if the
    # vendored version is 0..23, change _HOUR_OFFSET and this test together).
    card = render_input_card(
        tx_lat=0, tx_lon=0, rx_lat=10, rx_lon=10,
        month=1, hour_utc=0, ssn=10, freq_mhz=14.074, data_dir=Path("/d"),
    )
    assert "Path.hour 1" in card


def test_parse_report_extracts_bcr_and_snr():
    result = parse_report(FIXTURE.read_text())
    assert result == P533Result(reliability_pct=78.24, snr_db=-10.2)


def test_parse_report_raises_on_missing_columns():
    with pytest.raises(ValueError, match="BCR"):
        parse_report("no data here\n")


def test_p533_score_invokes_binary_and_parses(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        # Read the input-card contents now, while the temp dir p533_score
        # created is still alive — it is cleaned up before p533_score
        # returns, so the path itself is unreadable afterward.
        seen["card"] = Path(cmd[1]).read_text()
        # ITURHFProp usage: iturhfprop <input> <output> — write the report.
        Path(cmd[2]).write_text(FIXTURE.read_text())
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(p533.subprocess, "run", fake_run)
    result = p533.p533_score(
        tx_lat=35.0, tx_lon=-90.0, rx_lat=45.0, rx_lon=135.0,
        band="20m", month=7, hour=14, ssn=120.0,
    )
    assert result == P533Result(reliability_pct=78.24, snr_db=-10.2)
    assert "iturhfprop" in Path(seen["cmd"][0]).name
    assert "Path.frequency 14.0740" in seen["card"]


def test_p533_score_raises_on_nonzero_exit(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="boom")

    monkeypatch.setattr(p533.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="exit 3"):
        p533.p533_score(0, 0, 10, 10, "20m", 1, 0, 10.0)


@pytest.mark.skipif(not binary_path().exists(), reason="run `uv run build-p533` first")
def test_p533_score_against_real_binary():
    # A path that must be reliably open: 1000 km mid-latitude 20m, midday, high SSN.
    result = p533.p533_score(
        tx_lat=40.0, tx_lon=-100.0, rx_lat=40.0, rx_lon=-88.0,
        band="20m", month=7, hour=18, ssn=150.0,
    )
    assert 0.0 <= result.reliability_pct <= 100.0
    assert result.reliability_pct > 50.0
    assert -40.0 <= result.snr_db <= 60.0


SSN_FIXTURE = Path(__file__).parent.parent / "fixtures" / "swpc_solar_cycle_sample.json"


def test_ssn_by_month_prefers_smoothed_falls_back_to_observed(tmp_path, monkeypatch):
    def fake_get(url, **kwargs):
        assert url == p533.SWPC_SOLAR_CYCLE_URL
        class R:
            status_code = 200
            def json(self):
                return json.loads(SSN_FIXTURE.read_text())
            def raise_for_status(self):
                pass
        return R()

    monkeypatch.setattr(p533.httpx, "get", fake_get)
    out = p533.ssn_by_month(["2026-04", "2026-05"], cache_dir=tmp_path)
    assert out == {"2026-04": 133.0, "2026-05": 128.7}


def test_ssn_by_month_uses_cache_second_time(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        class R:
            status_code = 200
            def json(self):
                return json.loads(SSN_FIXTURE.read_text())
            def raise_for_status(self):
                pass
        return R()

    monkeypatch.setattr(p533.httpx, "get", fake_get)
    p533.ssn_by_month(["2026-04"], cache_dir=tmp_path)
    p533.ssn_by_month(["2026-04"], cache_dir=tmp_path)
    assert len(calls) == 1


def test_ssn_by_month_raises_on_unknown_month(tmp_path, monkeypatch):
    monkeypatch.setattr(
        p533, "_fetch_solar_cycle",
        lambda cache_dir: json.loads(SSN_FIXTURE.read_text()),
    )
    with pytest.raises(KeyError, match="1999-01"):
        p533.ssn_by_month(["1999-01"], cache_dir=tmp_path)


def _cells(rows):
    return pl.DataFrame(
        rows,
        schema={
            "window_start": pl.Datetime("us", "UTC"),
            "tx_field": pl.Utf8, "rx_field": pl.Utf8, "band": pl.Utf8,
        },
        orient="row",
    )


def test_predict_memoizes_by_hour(monkeypatch):
    calls = []

    def fake_score(tx_lat, tx_lon, rx_lat, rx_lon, band, month, hour, ssn):
        calls.append((band, month, hour))
        return P533Result(reliability_pct=80.0, snr_db=10.0)

    monkeypatch.setattr(p533, "p533_score", fake_score)
    model = P533Model(ssn_by_month={"2026-06": 120.0})
    ts = datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc)
    cells = _cells([
        (ts, "EM", "PM", "20m"),
        (ts.replace(minute=15), "EM", "PM", "20m"),   # same hour -> memo hit
        (ts.replace(minute=30), "EM", "PM", "20m"),
        (ts.replace(hour=15), "EM", "PM", "20m"),     # new hour -> new call
    ])
    out = model.predict(cells)
    assert out["p_open"].to_list() == [0.8, 0.8, 0.8, 0.8]
    assert len(calls) == 2  # 4 windows, 2 distinct (month,hour) keys
    assert out.columns == ["window_start", "tx_field", "rx_field", "band", "p_open"]
    assert len(out) == 4


def test_predict_abstains_when_ssn_month_missing(monkeypatch):
    monkeypatch.setattr(
        p533, "p533_score",
        lambda *a, **k: P533Result(reliability_pct=80.0, snr_db=10.0),
    )
    model = P533Model(ssn_by_month={"2026-06": 120.0})
    cells = _cells([
        (datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc), "EM", "PM", "20m"),
    ])
    out = model.predict(cells)
    assert out["p_open"].to_list() == [None]


def test_predict_abstains_on_failed_score(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("iturhfprop exit 1: ...")

    monkeypatch.setattr(p533, "p533_score", boom)
    model = P533Model(ssn_by_month={"2026-06": 120.0})
    cells = _cells([
        (datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc), "EM", "PM", "20m"),
    ])
    out = model.predict(cells)
    assert out["p_open"].to_list() == [None]
