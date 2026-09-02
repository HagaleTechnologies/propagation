"""Live PSKReporter MQTT accumulator (PRO-9). Unlike WSPRnet/RBN's monthly/
daily batch archives, PSKReporter has no public bulk-history download --
`mqtt.pskreporter.info` is the only way to get it, so this module
accumulates going forward rather than backfilling (ARCHITECTURE.md §3.2).

Connection/payload details verified against mqtt.pskreporter.info's own
documentation page (2026-09-02), not assumed:

    host: mqtt.pskreporter.info, TLS port 1884 (plain TCP: 1883)
    topic: pskr/filter/v2/{band}/{mode}/{tx}/{rx}/{tx_grid}/{rx_grid}/{tx_dxcc}/{rx_dxcc}
           ('+' = one segment wildcard, '#' = rest-of-topic wildcard; no auth)
    payload: one spot per message, JSON:
        sq (seq), f (freq Hz), md (mode), rp (SNR dB), t (report epoch),
        t_tx (normalized tx-start epoch), sc/sl (sender call/grid --
        the transmitting station, i.e. this repo's dx_call/dx_grid),
        rc/rl (receiver call/grid -- the reporting station, our
        de_call/de_grid), sa/ra (sender/receiver ADIF DXCC codes), b (band)

`t_tx` (not `t`) is used as `ts`: it's the transmission's nominal cycle
start, comparable across every station that reports the same transmission --
the same role WSPRnet's decode-window timestamp plays. Falls back to `t` if
`t_tx` is absent.

Resilience (PRO-9's "connection drop must not silently lose data" scenario):
paho-mqtt's `loop_forever()` auto-reconnects on a dropped TCP connection
(`reconnect_on_failure=True`, the default), and `on_connect` -- which
re-subscribes -- fires again after every reconnect, so a network blip
resumes streaming with no process restart. A *process* crash (not just a
dropped socket) still needs external supervision (launchd/systemd) to
restart the script -- that's a deployment concern, matching how
ARCHITECTURE.md §7 already plans launchd for M4's serving/score.py, not
something this module can guarantee on its own.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import paho.mqtt.client as mqtt
import polars as pl

from propagation.data.dedup import dedup_spots
from propagation.data.hygiene import is_qualifying_spot
from propagation.data.schema import SPOT_SCHEMA

logger = logging.getLogger(__name__)

PSKR_MQTT_HOST = "mqtt.pskreporter.info"
PSKR_MQTT_PORT_TLS = 1884
PSKR_MQTT_PORT_TCP = 1883


def pskr_topic(band: str = "+") -> str:
    """`band='+'` (default) subscribes to every band; pass a specific band
    (e.g. '20m') to narrow server-side instead of filtering client-side."""
    return f"pskr/filter/v2/{band}/#"


def parse_pskreporter_payload(payload: dict) -> dict | None:
    try:
        ts_epoch = payload.get("t_tx", payload.get("t"))
        ts = dt.datetime.fromtimestamp(int(ts_epoch), tz=dt.timezone.utc)
        freq_hz = int(payload["f"])
        snr = int(payload["rp"])
    except (KeyError, TypeError, ValueError):
        return None
    band = payload.get("b")
    mode = payload.get("md")
    dx_call = (payload.get("sc") or "").strip().upper()
    de_call = (payload.get("rc") or "").strip().upper()
    if not band or not mode or not dx_call or not de_call:
        return None
    return {
        "source": "pskreporter",
        "ts": ts,
        "band": band,
        "mode": mode.strip().upper(),
        "freq_hz": freq_hz,
        "dx_call": dx_call,
        "de_call": de_call,
        "dx_grid": (payload.get("sl") or None),
        "de_grid": (payload.get("rl") or None),
        "dx_lat": None,
        "dx_lon": None,
        "de_lat": None,
        "de_lon": None,
        "snr_db": snr,
        "tx_dbm": None,
    }


@dataclass
class FlushResult:
    hour: dt.datetime
    path: Path
    n_received: int
    n_parsed: int
    n_qualifying: int
    rejection_counts: dict[str, int] = field(default_factory=dict)


class PSKReporterAccumulator:
    """Buffers parsed spots in memory and flushes one hourly Parquet file per
    UTC hour boundary crossed (matching ARCHITECTURE.md's lake partitioning
    and PRO-9's "same on-disk format as the batch WSPRnet extracts"
    requirement) -- QA/hygiene filtering and dedup_spots run once per flush,
    same as the batch extractors, not per-message.

    `on_flush` (default: write_hourly_parquet) is injectable so a caller can
    redirect output or observe flushes in tests without touching disk or a
    real MQTT broker -- construct with `client_factory` to inject a fake
    paho.mqtt.client.Client-alike for the same reason.
    """

    def __init__(
        self,
        out_dir: Path,
        band: str = "+",
        client_factory: Callable[[], mqtt.Client] | None = None,
        on_flush: Callable[[list[dict], dt.datetime, Path], FlushResult] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.band = band
        self._buffer: list[dict] = []
        self._buffer_hour: dt.datetime | None = None
        self._on_flush = on_flush or write_hourly_parquet
        self.flush_results: list[FlushResult] = []
        self._client = (client_factory or self._default_client)()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

    def _default_client(self) -> mqtt.Client:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.tls_set()
        return client

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        topic = pskr_topic(self.band)
        logger.info("connected (reason_code=%s), subscribing to %s", reason_code, topic)
        client.subscribe(topic)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None) -> None:
        logger.warning("disconnected (reason_code=%s) -- loop_forever will auto-reconnect", reason_code)

    def _on_message(self, client, userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        self.ingest(payload)

    def ingest(self, payload: dict) -> None:
        """Handle one decoded JSON payload -- also the entry point tests use
        to drive the accumulator without a real MQTT message object."""
        ts_epoch = payload.get("t_tx", payload.get("t"))
        try:
            hour = dt.datetime.fromtimestamp(int(ts_epoch), tz=dt.timezone.utc).replace(
                minute=0, second=0, microsecond=0
            )
        except (TypeError, ValueError):
            hour = None

        if self._buffer_hour is not None and hour is not None and hour != self._buffer_hour:
            self.flush()
        if self._buffer_hour is None:
            self._buffer_hour = hour
        self._buffer.append(payload)

    def flush(self) -> FlushResult | None:
        if not self._buffer:
            return None
        hour = self._buffer_hour or dt.datetime.now(dt.timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        out_path = self.out_dir / f"pskreporter-{hour:%Y-%m-%dT%H}.parquet"
        result = self._on_flush(self._buffer, hour, out_path)
        self.flush_results.append(result)
        self._buffer = []
        self._buffer_hour = None
        return result

    def run_forever(self) -> None:  # pragma: no cover -- network loop, not unit-tested
        self._client.connect(PSKR_MQTT_HOST, PSKR_MQTT_PORT_TLS)
        try:
            self._client.loop_forever()
        finally:
            self.flush()


def write_hourly_parquet(raw_payloads: list[dict], hour: dt.datetime, out_path: Path) -> FlushResult:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rejection_counts: dict[str, int] = {}
    n_parsed = 0
    rows: list[dict] = []
    for payload in raw_payloads:
        parsed = parse_pskreporter_payload(payload)
        if parsed is None:
            continue
        n_parsed += 1
        ok, reason = is_qualifying_spot(parsed)
        if not ok:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        rows.append(parsed)

    if not rows:
        spots = pl.DataFrame(schema=SPOT_SCHEMA)
    else:
        spots = pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")})
        for col in SPOT_SCHEMA:
            if col not in spots.columns:
                spots = spots.with_columns(pl.lit(None).alias(col))
        spots = dedup_spots(spots)

    if out_path.exists():
        existing = pl.read_parquet(out_path)
        spots = dedup_spots(pl.concat([existing, spots], how="vertical_relaxed"))
    spots.write_parquet(out_path)

    return FlushResult(
        hour=hour,
        path=out_path,
        n_received=len(raw_payloads),
        n_parsed=n_parsed,
        n_qualifying=spots.height,
        rejection_counts=rejection_counts,
    )
