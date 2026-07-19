from datetime import datetime, timezone
from pathlib import Path

import pytest

from propagation.data import spaceweather

FIXTURE = Path(__file__).parent.parent / "fixtures" / "omni2_sample.dat"


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def test_parse_omni2_extracts_the_right_columns():
    df = spaceweather._parse_omni2(FIXTURE.read_text(), year=2014)
    assert df.columns == ["time", "kp", "f107", "bz_gsm", "solar_wind_speed", "dst"]
    assert df["time"].to_list() == [_utc(2014, 1, 1, 0), _utc(2014, 1, 1, 1), _utc(2014, 1, 1, 2)]
    # word 39 (Kp) on row 0 is "7" -> tens digit=0, units 7=("-" tier) -> 0.667... see decode below
    assert df["f107"].to_list() == pytest.approx([154.3, 154.3, 154.3])
    assert df["bz_gsm"].to_list() == pytest.approx([0.6, -0.2, -1.8])
    assert df["solar_wind_speed"].to_list() == pytest.approx([399.0, 395.0, 386.0])
    assert df["dst"].to_list() == pytest.approx([4.0, 3.0, 1.0])


def test_decode_omni2_kp():
    # OMNI2 Kp is coded as an integer: tens digit = whole Kp, units digit
    # in {0,3,7} = {"-", "o"/nothing, "+"} i.e. 33 = "3+" = 3.333, 40 = "4o" = 4.0,
    # 57 = "5+" = 5.333... this repo's own Kp convention (see
    # propagation.eval.stratify._parse_gfz) uses thirds (x.0/x.333/x.667);
    # OMNI2's units digit encodes the same thirds on a 0-9 scale (0,3,7=+/-,o).
    assert spaceweather._decode_omni2_kp(33) == pytest.approx(3.333, abs=0.01)
    assert spaceweather._decode_omni2_kp(40) == pytest.approx(4.0, abs=0.01)
    assert spaceweather._decode_omni2_kp(57) == pytest.approx(5.667, abs=0.01)


def test_fetch_omni2_year_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        class R:
            status_code = 200
            text = FIXTURE.read_text()
            def raise_for_status(self):
                pass
        return R()

    monkeypatch.setattr(spaceweather.httpx, "get", fake_get)
    spaceweather.fetch_omni2_year(2014, cache_dir=tmp_path)
    df = spaceweather.fetch_omni2_year(2014, cache_dir=tmp_path)
    assert len(calls) == 1
    assert len(df) == 3


def test_fetch_omni2_range_concatenates_years(tmp_path, monkeypatch):
    def fake_get(url, **kwargs):
        class R:
            status_code = 200
            text = FIXTURE.read_text()
            def raise_for_status(self):
                pass
        return R()

    monkeypatch.setattr(spaceweather.httpx, "get", fake_get)
    df = spaceweather.fetch_omni2_range(2014, 2015, cache_dir=tmp_path)
    assert len(df) == 6  # 3 rows/year fixture x 2 years
