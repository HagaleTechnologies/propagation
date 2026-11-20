"""PRO-10 cqdx R2 backfill command tests (AC1 pagination, AC2 key bounds,
AC3 credentials + atomic publish). Exercises the real command with a fake
S3 boundary: paging, gzip parsing and the Parquet writer are the real ones."""
import contextlib
import datetime as dt
import gzip
import io
import json
import re
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from extract_cqdx import (  # noqa: E402
    _ARCHIVE_ACCESS_MSG,
    _MISSING_CREDS_MSG,
    _NO_SPOTS_MSG,
    run_backfill,
)
from propagation.data.schema import SPOT_SCHEMA  # noqa: E402

GOOD_ACCOUNT = "0123456789abcdef0123456789abcdef"
_CREDS = {
    "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
    "AWS_SECRET_ACCESS_KEY": "secret",
    "AWS_SESSION_TOKEN": "session-token",
}

_TS = 1780000000  # 2026-05-29T07:06:40Z


def _ndjson_object(rows: list[dict]) -> bytes:
    lines = "\n".join(json.dumps(r) for r in rows) + "\n"
    return gzip.compress(lines.encode())


def _payload(dx_call="K1BBB", ts=_TS, source="pskreporter", **overrides):
    row = {
        "timestamp": ts,
        "frequency": 14074000,
        "band": "20m",
        "mode": "ft8",
        "dxCall": dx_call,
        "deCall": "W1AAA",
        "dxGrid": "DM79",
        "deGrid": "FN31",
        "snr": -10,
        "source": source,
    }
    row.update(overrides)
    return row


class FakeR2:
    """Implements only list_objects_v2 (paginated) and get_object."""

    def __init__(self, objects: dict[str, bytes], page_size=2, fail_keys=()):
        self.objects = objects
        self.page_size = page_size
        self.fail_keys = set(fail_keys)
        self.lists: list[tuple[str, str]] = []
        self.gets: list[str] = []
        self.created = 0

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None, **kwargs):
        self.lists.append((Bucket, Prefix))
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        token = ContinuationToken
        if token:
            keys = [k for k in keys if k > token]
        page = keys[: self.page_size]
        truncated = len(keys) > len(page)
        return {
            "Contents": [{"Key": k} for k in page],
            "IsTruncated": truncated,
            "NextContinuationToken": page[-1] if truncated else None,
        }

    def get_object(self, Bucket, Key, **kwargs):
        self.gets.append(Key)
        if Key in self.fail_keys:
            raise RuntimeError("AccessDenied")
        if Key not in self.objects:
            raise RuntimeError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[Key])}


def _run(fake, tmp_path, *, creds=_CREDS, start="2026-06-01", end="2026-06-02",
         out_name="cqdx-june", account=GOOD_ACCOUNT):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = run_backfill(
            account_id=account,
            start_date=dt.datetime.strptime(start, "%Y-%m-%d").date(),
            end_date=dt.datetime.strptime(end, "%Y-%m-%d").date(),
            out_dir=tmp_path / out_name,
            credentials=creds,
            client_factory=lambda: (fake.__setattr__("created", fake.created + 1), fake)[1],
        )
    return code, stdout.getvalue(), stderr.getvalue()


def test_backfill_writes_shared_schema_from_paginated_r2(tmp_path):
    objects = {
        "raw/2026-06-01/05/01ARZ3NDEKTSV4RRFFQ69G5FAV.ndjson.gz": _ndjson_object([
            _payload(dx_call="K1BBB"),
            _payload(dx_call="K2CCC"),
        ]),
        "raw/2026-06-01/06/01ARZ3NDEKTSV4RRFFQ69G5FAW.ndjson.gz": _ndjson_object([
            # exact duplicate of a transmission already seen in the 05 object:
            # dedup_spots must collapse it across the day's objects
            _payload(dx_call="K1BBB", ts=_TS),
            _payload(dx_call="K3CCC", ts=_TS + 900, source="rbn"),
            # manual + malformed rows are counted, not written
            _payload(dx_call="K4CCC", source="user"),
        ]),
        "raw/2026-06-01/07/01ARZ3NDEKTSV4RRFFQ69G5FAX.ndjson.gz": _ndjson_object([
            _payload(dx_call="K5CCC", ts=_TS + 1800),
        ]),
    }
    fake = FakeR2(objects, page_size=1)  # force multi-page listings
    code, out, err = _run(fake, tmp_path)
    assert code == 0, (out, err)
    parquet = tmp_path / "cqdx-june" / "cqdx-2026-06-01.parquet"
    frame = pl.read_parquet(parquet)
    assert frame.schema == pl.DataFrame(schema=SPOT_SCHEMA).schema
    assert sorted(frame["dx_call"].to_list()) == ["K1BBB", "K2CCC", "K3CCC", "K5CCC"]
    assert set(frame["source"].unique().to_list()) == {"cqdx"}
    # pagination really happened: content pages + one terminating call at page_size=1
    assert len(fake.lists) >= 3
    match = re.search(r"Wrote (\d+) cqdx spots to .+ \((\d+) day\(s\), (\d+) object\(s\), (\d+) rejected\)", out)
    assert match and match.groups() == ("4", "1", "3", "1"), out


def test_backfill_never_fetches_unexpected_keys(tmp_path):
    objects = {
        "raw/2026-06-01/05/01ARZ3NDEKTSV4RRFFQ69G5FAV.ndjson.gz": _ndjson_object([_payload()]),
        # misbehaving listing entries -- every one of these must be skipped
        "raw/2026-05-31/23/01ARZ3NDEKTSV4RRFFQ69G5FAZ.ndjson.gz": _ndjson_object([_payload(dx_call="KEEPM3")]),
        "raw/2026-06-02/01/01ARZ3NDEKTSV4RRFFQ69G5FB0.ndjson.gz": _ndjson_object([_payload(dx_call="KEEPM3")]),
        "raw/2026-06-01/05/not-a-ulid-key.ndjson.gz": _ndjson_object([_payload(dx_call="KEEPM3")]),
        "raw/2026-06-01/05/01ARZ3NDEKTSV4RRFFQ69G5FAV.txt": _ndjson_object([_payload(dx_call="KEEPM3")]),
        "pending-imports/2026-06-01/logbook.ndjson.gz": _ndjson_object([_payload(dx_call="KEEPM3")]),
    }
    fake = FakeR2(objects)
    # the fake lists by prefix, so inject the foreign keys into the day listing
    original = fake.list_objects_v2

    def leaky_list(Bucket, Prefix, ContinuationToken=None, **kw):
        page = original(Bucket, Prefix, ContinuationToken, **kw)
        if Prefix == "raw/2026-06-01/":
            page["Contents"].extend({"Key": k} for k in (
                "raw/2026-05-31/23/01ARZ3NDEKTSV4RRFFQ69G5FAZ.ndjson.gz",
                "raw/2026-06-02/01/01ARZ3NDEKTSV4RRFFQ69G5FB0.ndjson.gz",
                "raw/2026-06-01/05/not-a-ulid-key.ndjson.gz",
                "raw/2026-06-01/05/01ARZ3NDEKTSV4RRFFQ69G5FAV.txt",
                "pending-imports/2026-06-01/logbook.ndjson.gz",
                "raw/2026-06-01/99/01ARZ3NDEKTSV4RRFFQ69G5FAV.ndjson.gz",
            ))
        return page

    fake.list_objects_v2 = leaky_list
    code, out, err = _run(fake, tmp_path)
    assert code == 0, (out, err)
    assert fake.gets == ["raw/2026-06-01/05/01ARZ3NDEKTSV4RRFFQ69G5FAV.ndjson.gz"]
    assert all(bucket == "cqdx-archive" and prefix.startswith("raw/2026-06-01/") for bucket, prefix in fake.lists)
    frame = pl.read_parquet(tmp_path / "cqdx-june" / "cqdx-2026-06-01.parquet")
    assert frame["dx_call"].to_list() == ["K1BBB"]


def test_backfill_requires_explicit_temporary_credentials(tmp_path):
    objects = {"raw/2026-06-01/05/01ARZ3NDEKTSV4RRFFQ69G5FAV.ndjson.gz": _ndjson_object([_payload()])}
    for missing in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        fake = FakeR2(objects)
        creds = {k: v for k, v in _CREDS.items() if k != missing}
        code, out, err = _run(fake, tmp_path, creds=creds, out_name=f"out-{missing}")
        assert code == 2
        assert _MISSING_CREDS_MSG in err
        assert fake.created == 0  # exit happens before any R2 client exists
        assert fake.lists == [] and fake.gets == []
        assert not (tmp_path / f"out-{missing}").exists()
    # a long-lived key pair without the session token is equally refused
    fake = FakeR2(objects)
    code, _, err = _run(
        fake, tmp_path, creds={"AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b", "AWS_SESSION_TOKEN": ""},
        out_name="out-empty-token",
    )
    assert code == 2 and _MISSING_CREDS_MSG in err


def test_backfill_read_failure_does_not_publish_partial_output(tmp_path):
    day1 = "raw/2026-06-01/05/01ARZ3NDEKTSV4RRFFQ69G5FAV.ndjson.gz"
    day2 = "raw/2026-06-02/05/01ARZ3NDEKTSV4RRFFQ69G5FAY.ndjson.gz"
    objects = {
        day1: _ndjson_object([_payload(dx_call="K1BBB", ts=_TS)]),
        day2: _ndjson_object([_payload(dx_call="K2CCC", ts=_TS + 86400)]),
    }
    fake = FakeR2(objects, fail_keys={day2})
    code, out, err = _run(fake, tmp_path, start="2026-06-01", end="2026-06-03")
    assert code == 1
    assert _ARCHIVE_ACCESS_MSG in err
    out_dir = tmp_path / "cqdx-june"
    assert not out_dir.exists()  # nothing published on failure
    assert not (tmp_path / ".cqdx-june.staging").exists()  # staging cleaned
    assert day1 in fake.gets  # first day was read, then the run aborted

    # clean retry after the failure heals publishes the full range
    fake_retry = FakeR2(objects)
    code, out, err = _run(fake_retry, tmp_path, start="2026-06-01", end="2026-06-03")
    assert code == 0, err
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "cqdx-2026-06-01.parquet", "cqdx-2026-06-02.parquet",
    ]


def test_backfill_empty_range_exits_nonzero_and_publishes_nothing(tmp_path):
    fake = FakeR2({})  # archive has nothing for the requested day
    code, out, err = _run(fake, tmp_path)
    assert code == 1
    assert _NO_SPOTS_MSG in err
    assert not (tmp_path / "cqdx-june").exists()


def test_backfill_refuses_existing_snapshot_and_bad_account(tmp_path):
    objects = {"raw/2026-06-01/05/01ARZ3NDEKTSV4RRFFQ69G5FAV.ndjson.gz": _ndjson_object([_payload()])}
    fake = FakeR2(objects)
    (tmp_path / "cqdx-june").mkdir()
    code, _, err = _run(fake, tmp_path)
    assert code == 2 and "refusing to overwrite" in err

    with pytest.raises(SystemExit) as excinfo, contextlib.redirect_stderr(io.StringIO()) as err:
        run_backfill(
            account_id="not-hex",
            start_date=dt.date(2026, 6, 1),
            end_date=dt.date(2026, 6, 2),
            out_dir=tmp_path / "other",
            credentials=_CREDS,
            client_factory=lambda: FakeR2(objects),
        )
    assert excinfo.value.code == 2
    assert "account-id" in err.getvalue()
