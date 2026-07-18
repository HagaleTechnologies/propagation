from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from propagation.eval import stratify

FIXTURE = Path(__file__).parent.parent / "fixtures" / "gfz_kp_sample.txt"


def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_parse_gfz_kp():
    df = stratify._parse_gfz(FIXTURE.read_text())
    assert df.columns == ["block_start", "kp"]
    assert df["block_start"].to_list() == [
        _utc(2026, 6, 15, 0), _utc(2026, 6, 15, 3),
        _utc(2026, 6, 15, 6), _utc(2026, 6, 15, 9),
    ]
    assert df["kp"].to_list() == [1.667, 2.0, 5.333, 6.0]


def test_parse_gfz_excludes_provisional_rows():
    """The fixture's 5th line has D=0 (provisional) and must be dropped."""
    df = stratify._parse_gfz(FIXTURE.read_text())
    assert len(df) == 4
    assert _utc(2026, 6, 15, 12) not in df["block_start"].to_list()


def test_parse_gfz_raises_clear_error_on_too_few_fields():
    bad = "2026 06 15 00.0 01.5 34500.00000 34500.06250  1.667     6\n"
    with pytest.raises(ValueError, match="GFZ"):
        stratify._parse_gfz(bad)


def test_parse_gfz_raises_clear_error_on_non_numeric_field():
    bad = "2026 06 15 00.0 01.5 34500.00000 34500.06250  bogus     6 1\n"
    with pytest.raises(ValueError, match="GFZ"):
        stratify._parse_gfz(bad)


def test_fetch_definitive_kp_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        class R:
            status_code = 200
            text = FIXTURE.read_text()
            def raise_for_status(self):
                pass
        return R()

    monkeypatch.setattr(stratify.httpx, "get", fake_get)
    stratify.fetch_definitive_kp(cache_dir=tmp_path)
    df = stratify.fetch_definitive_kp(cache_dir=tmp_path)
    assert len(calls) == 1
    assert len(df) == 4


def test_tag_storm_windows_joins_3h_blocks():
    kp = stratify._parse_gfz(FIXTURE.read_text())
    labels = pl.DataFrame(
        {
            "window_start": [_utc(2026, 6, 15, 2, 45), _utc(2026, 6, 15, 7, 15)],
            "open": [1, 0],
        },
        schema_overrides={"window_start": pl.Datetime("us", "UTC")},
    )
    out = stratify.tag_storm_windows(labels, kp)
    assert out["kp"].to_list() == [1.667, 5.333]
    assert out["is_storm"].to_list() == [False, True]
