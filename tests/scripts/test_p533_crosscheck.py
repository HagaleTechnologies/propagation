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
    p533_crosscheck.run(n=3, out_path=out)
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "tx_lat,tx_lon,rx_lat,rx_lon,band,month,hour,ssn,reliability_pct,snr_db"
    assert len(lines) == 4
