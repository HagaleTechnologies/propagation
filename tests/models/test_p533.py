import subprocess
from pathlib import Path

import pytest

from propagation.data.schema import SUPPORTED_BANDS
from propagation.models import p533
from propagation.models.p533 import BAND_FREQ_MHZ, P533Result, parse_report, render_input_card
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
    card = Path(seen["cmd"][1]).read_text()
    assert "Path.frequency 14.0740" in card


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
