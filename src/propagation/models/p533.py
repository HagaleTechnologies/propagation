"""ITU-R P.533 baseline: wraps the vendored ITURHFProp binary (baselines/p533).

Scores field-center -> field-center paths (ARCHITECTURE.md sec 5 M-1,
midpoint-to-midpoint), reusing propagation.data.geo.grid_to_latlon for the
field-center math. Standalone: no cqdx dependency (README boundaries).
"""
from __future__ import annotations

# Primary digital-activity (FT8) dial frequency per band, MHz — the
# frequency at which most label-generating spots occur, hence the honest
# frequency to score P.533 at.
BAND_FREQ_MHZ: dict[str, float] = {
    "160m": 1.840, "80m": 3.573, "60m": 5.357, "40m": 7.074,
    "30m": 10.136, "20m": 14.074, "17m": 18.100, "15m": 21.074,
    "12m": 24.915, "10m": 28.074, "6m": 50.313,
}
