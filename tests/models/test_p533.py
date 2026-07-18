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
