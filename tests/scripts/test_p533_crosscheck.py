import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
import p533_crosscheck  # noqa: E402
from propagation.models.p533 import P533Result


def test_sample_paths_is_deterministic_and_sized():
    a = p533_crosscheck.sample_paths(n=100)
    b = p533_crosscheck.sample_paths(n=100)
    assert a == b
    assert len(a) == 100
    assert len(set(a)) == 100


def test_write_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        p533_crosscheck, "p533_score",
        lambda **kw: P533Result(reliability_pct=50.0, snr_db=5.0),
    )
    out = tmp_path / "x.csv"
    skipped = p533_crosscheck.run(n=3, out_path=out)
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "tx_lat,tx_lon,rx_lat,rx_lon,band,month,hour,ssn,reliability_pct,snr_db"
    assert len(lines) == 4
    assert skipped == 0


def test_write_csv_skips_failing_paths_instead_of_aborting(tmp_path, monkeypatch):
    # A path outside P.533's valid domain (e.g. 6m, out of HF range) raises;
    # the run must skip it and keep going, not abort the whole ~n-path batch.
    calls = {"n": 0}

    def flaky_score(**kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("band '6m' (50.313 MHz) is outside ITU-R P.533's valid frequency range")
        return P533Result(reliability_pct=50.0, snr_db=5.0)

    monkeypatch.setattr(p533_crosscheck, "p533_score", flaky_score)
    out = tmp_path / "x.csv"
    skipped = p533_crosscheck.run(n=3, out_path=out)
    lines = out.read_text().strip().splitlines()
    assert skipped == 1
    assert len(lines) == 3  # header + 2 successful rows (the 3rd sample failed)
