from propagation.data.schema import SUPPORTED_BANDS
from propagation.models.p533 import BAND_FREQ_MHZ


def test_every_supported_band_has_a_frequency():
    assert set(BAND_FREQ_MHZ) == SUPPORTED_BANDS


def test_frequencies_are_the_primary_ft8_dial_frequency():
    expected = {
        "160m": 1.840, "80m": 3.573, "60m": 5.357, "40m": 7.074,
        "30m": 10.136, "20m": 14.074, "17m": 18.100, "15m": 21.074,
        "12m": 24.915, "10m": 28.074, "6m": 50.313,
    }
    assert BAND_FREQ_MHZ == expected


from pathlib import Path

from propagation.models.p533 import P533Result, parse_report, render_input_card

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
    assert result == P533Result(reliability_pct=87.0, snr_db=23.0)


def test_parse_report_raises_on_missing_columns():
    import pytest
    with pytest.raises(ValueError, match="BCR"):
        parse_report("no data here\n")
