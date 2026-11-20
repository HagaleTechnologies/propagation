"""PRO-10: optional backfill from cqdx's private Cloudflare R2 archive.

Reads `raw/YYYY-MM-DD/HH/{ulid}.ndjson.gz` objects from the `cqdx-archive`
bucket (dispensa Q-0062) using read-only temporary S3 credentials scoped to
`raw/`, and writes one `cqdx-YYYY-MM-DD.parquet` per non-empty archive day
into a fresh output directory. The archive is an optional accelerant: the
public pipeline never needs this command, boto3 lives behind the
`cqdx-r2` extra, and nothing here runs from the default dependency set.

Usage:
    uv run --extra cqdx-r2 python scripts/extract_cqdx.py \
        --account-id 0123456789abcdef0123456789abcdef \
        --start-date 2026-06-01 --end-date 2026-06-02 --out-dir data/cqdx-june

`--start-date` is inclusive, `--end-date` exclusive, in UTC archive
partition dates. Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and
AWS_SESSION_TOKEN in the environment (a session token is mandatory: these
are temporary scoped credentials, never long-lived keys). See
docs/cqdx-backfill.md for the owner-side provisioning story.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import sys
import zlib
from pathlib import Path
from typing import Callable

from propagation.data.cqdx_r2 import (
    archive_rows_to_frame,
    iter_requested_days,
    parse_archive_object,
)
from propagation.data.dedup import dedup_spots

CQDX_BUCKET = "cqdx-archive"
R2_REGION = "auto"

_ACCOUNT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
# ULID alphabet: Crockford base32, which excludes I, L, O and U.
_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
_OBJECT_KEY_RE = re.compile(r"^raw/(\d{4})-(\d{2})-(\d{2})/(\d{2})/([0-9A-HJKMNP-TV-Z]{26})\.ndjson\.gz$")

_MISSING_CREDS_MSG = (
    "Error: cqdx backfill requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
    "and AWS_SESSION_TOKEN for read-only raw/ access."
)
_ARCHIVE_ACCESS_MSG = (
    "Error: cqdx archive access failed. Check that the temporary credential "
    "can list and read raw/."
)
_NO_SPOTS_MSG = "Error: no qualifying cqdx spots in the requested archive dates."


class ArchiveAccessError(Exception):
    """Any R2 transport/service failure (list or get). Aborts the run."""


def _r2_endpoint_url(account_id: str) -> str:
    return f"https://{account_id}.r2.cloudflarestorage.com"


def _validate_account_id(account_id: str) -> str:
    if not _ACCOUNT_ID_RE.match(account_id or ""):
        print(
            "Error: --account-id must be exactly 32 lowercase hex characters "
            "(the Cloudflare R2 account ID).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return account_id


def _day_prefix(day: dt.date) -> str:
    return f"raw/{day:%Y-%m-%d}/"


def _validated_object_key(key: str, expected_day: dt.date) -> str | None:
    """The exact key to GET, or None when a listed key does not belong to
    the requested day's `raw/YYYY-MM-DD/HH/{ulid}.ndjson.gz` layout. The
    listing is data we received, not data we requested: anything that does
    not match the strict shape is skipped, never fetched."""
    m = _OBJECT_KEY_RE.match(key)
    if m is None:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hour = m.group(4)
    if (y, mo, d) != (expected_day.year, expected_day.month, expected_day.day):
        return None
    if not (0 <= int(hour) <= 23):
        return None
    return key


def _list_day_keys(client, day: dt.date) -> list[str]:
    """Paginated ListObjectsV2 restricted to the requested day's prefix.
    Never a bucket-root listing -- Prefix is always `raw/YYYY-MM-DD/`."""
    keys: list[str] = []
    token: str | None = None
    while True:
        try:
            kwargs: dict = {"Bucket": CQDX_BUCKET, "Prefix": _day_prefix(day)}
            if token is not None:
                kwargs["ContinuationToken"] = token
            page = client.list_objects_v2(**kwargs)
        except Exception as exc:  # boto3 ClientError / EndpointConnectionError
            raise ArchiveAccessError(str(exc)) from exc
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key")
            if key is None:
                continue
            validated = _validated_object_key(key, day)
            if validated is not None:
                keys.append(validated)
        if not page.get("IsTruncated"):
            return keys
        token = page.get("NextContinuationToken")
        if not token:
            return keys


def _get_object(client, key: str) -> bytes:
    try:
        resp = client.get_object(Bucket=CQDX_BUCKET, Key=key)
        return resp["Body"].read()
    except Exception as exc:
        raise ArchiveAccessError(str(exc)) from exc


def run_backfill(
    account_id: str,
    start_date: dt.date,
    end_date: dt.date,
    out_dir: Path,
    credentials: dict[str, str | None],
    client_factory: Callable[[], object] | None = None,
    stdout=None,
    stderr=None,
) -> int:
    """Run one bounded backfill. Returns the process exit code."""
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    _validate_account_id(account_id)
    if start_date >= end_date:
        print("Error: --start-date must be before --end-date.", file=stderr)
        return 2

    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        if not credentials.get(var):
            print(_MISSING_CREDS_MSG, file=stderr)
            return 2

    out_dir = Path(out_dir)
    if out_dir.exists():
        print(
            f"Error: refusing to overwrite an existing snapshot at {out_dir}. "
            "Pass a fresh output path.",
            file=stderr,
        )
        return 2
    staging = out_dir.parent / f".{out_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)

    if client_factory is None:
        def client_factory() -> object:
            import boto3  # optional `cqdx-r2` extra; never imported otherwise

            return boto3.client(
                "s3",
                endpoint_url=_r2_endpoint_url(account_id),
                aws_access_key_id=credentials["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=credentials["AWS_SECRET_ACCESS_KEY"],
                aws_session_token=credentials["AWS_SESSION_TOKEN"],
                region_name=R2_REGION,
            )

    try:
        client = client_factory()
    except SystemExit:
        raise
    except Exception as exc:
        print(_ARCHIVE_ACCESS_MSG, file=stderr)
        print(f"client creation failed: {type(exc).__name__}", file=stderr)
        return 1

    total_rows = 0
    n_days = 0
    n_objects = 0
    n_rejected = 0
    rejection_totals: dict[str, int] = {}
    try:
        staging.mkdir(parents=True, exist_ok=True)
        for day in iter_requested_days(start_date, end_date):
            keys = _list_day_keys(client, day)
            day_rows: list[dict] = []
            for key in keys:
                body = _get_object(client, key)
                n_objects += 1
                parsed = parse_archive_object(body)
                day_rows.extend(parsed.rows)
                n_rejected += sum(parsed.rejection_counts.values())
                for reason, count in parsed.rejection_counts.items():
                    rejection_totals[reason] = rejection_totals.get(reason, 0) + count
            if not day_rows:
                print(f"no qualifying rows in archive day {day}", file=stderr)
                continue
            frame = dedup_spots(archive_rows_to_frame(day_rows))
            day_path = staging / f"cqdx-{day:%Y-%m-%d}.parquet"
            frame.write_parquet(day_path)
            total_rows += frame.height
            n_days += 1
            print(f"day {day}: {frame.height} qualifying spots ({len(keys)} objects)", file=stderr)
    except ArchiveAccessError:
        shutil.rmtree(staging, ignore_errors=True)
        print(_ARCHIVE_ACCESS_MSG, file=stderr)
        return 1
    except (OSError, EOFError, zlib.error):
        # truncated/corrupt gzip raises from parse_archive_object's GzipFile
        # (EOFError: truncated stream; zlib.error: corrupt deflate; OSError:
        # BadGzipFile and friends)
        shutil.rmtree(staging, ignore_errors=True)
        print(_ARCHIVE_ACCESS_MSG, file=stderr)
        return 1

    if total_rows == 0:
        shutil.rmtree(staging, ignore_errors=True)
        print(_NO_SPOTS_MSG, file=stderr)
        return 1

    if rejection_totals:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(rejection_totals.items()))
        print(f"rejected rows by reason: {summary}", file=stderr)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(out_dir)
    print(f"Wrote {total_rows} cqdx spots to {out_dir} ({n_days} day(s), {n_objects} object(s), {n_rejected} rejected).")
    return 0


def _parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PRO-10 optional cqdx R2 archive backfill")
    ap.add_argument("--account-id", required=True, help="Cloudflare R2 account ID (32 hex chars)")
    ap.add_argument("--start-date", required=True, type=_parse_date, help="UTC archive date, inclusive (YYYY-MM-DD)")
    ap.add_argument("--end-date", required=True, type=_parse_date, help="UTC archive date, exclusive (YYYY-MM-DD)")
    ap.add_argument("--out-dir", required=True, type=Path, help="fresh destination directory for daily Parquet")
    args = ap.parse_args(argv)
    credentials = {
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_SESSION_TOKEN": os.environ.get("AWS_SESSION_TOKEN"),
    }
    return run_backfill(
        account_id=args.account_id,
        start_date=args.start_date,
        end_date=args.end_date,
        out_dir=args.out_dir,
        credentials=credentials,
    )


if __name__ == "__main__":
    raise SystemExit(main())
