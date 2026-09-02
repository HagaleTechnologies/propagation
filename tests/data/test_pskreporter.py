import datetime as dt

import polars as pl

from propagation.data.pskreporter import (
    FlushResult,
    PSKReporterAccumulator,
    parse_pskreporter_payload,
    pskr_topic,
    write_hourly_parquet,
)

REAL_PAYLOAD = {
    "sq": 30142870791, "f": 21074653, "md": "FT8", "rp": -5,
    "t": 1662407712, "t_tx": 1662407697,
    "sc": "SP2EWQ", "sl": "JO93fn42", "rc": "CU3AT", "rl": "HM68jp36",
    "sa": 269, "ra": 149, "b": "15m",
}


def test_pskr_topic_wildcards():
    assert pskr_topic() == "pskr/filter/v2/+/#"
    assert pskr_topic("20m") == "pskr/filter/v2/20m/#"


def test_parse_pskreporter_payload_maps_fields():
    row = parse_pskreporter_payload(REAL_PAYLOAD)
    assert row["source"] == "pskreporter"
    assert row["band"] == "15m"
    assert row["mode"] == "FT8"
    assert row["dx_call"] == "SP2EWQ"
    assert row["de_call"] == "CU3AT"
    assert row["dx_grid"] == "JO93fn42"
    assert row["de_grid"] == "HM68jp36"
    assert row["snr_db"] == -5
    assert row["freq_hz"] == 21074653
    assert row["tx_dbm"] is None
    # t_tx (normalized), not t, is the canonical timestamp
    assert row["ts"] == dt.datetime.fromtimestamp(1662407697, tz=dt.timezone.utc)


def test_parse_pskreporter_payload_falls_back_to_t_without_t_tx():
    payload = dict(REAL_PAYLOAD)
    del payload["t_tx"]
    row = parse_pskreporter_payload(payload)
    assert row["ts"] == dt.datetime.fromtimestamp(1662407712, tz=dt.timezone.utc)


def test_parse_pskreporter_payload_rejects_missing_fields():
    assert parse_pskreporter_payload({"sq": 1}) is None


def test_write_hourly_parquet_applies_hygiene_and_dedup(tmp_path):
    self_spot = dict(REAL_PAYLOAD, sc="SP2EWQ", rc="SP2EWQ")  # same base call -> self_spot
    out_path = tmp_path / "pskreporter-2022-09-05T20.parquet"
    result = write_hourly_parquet([REAL_PAYLOAD, self_spot], hour=dt.datetime(2022, 9, 5, 20, tzinfo=dt.timezone.utc), out_path=out_path)
    assert result.n_received == 2
    assert result.n_parsed == 2
    assert result.rejection_counts.get("self_spot") == 1
    assert result.n_qualifying == 1
    assert out_path.exists()
    spots = pl.read_parquet(out_path)
    assert spots.height == 1
    assert spots["source"][0] == "pskreporter"


def test_write_hourly_parquet_merges_into_existing_file(tmp_path):
    # Simulates a process restart mid-hour: a second flush to the same hourly
    # file must merge+dedup with what's already on disk, not overwrite it --
    # this is what makes "resume without manual restart" actually not lose
    # already-flushed data.
    out_path = tmp_path / "pskreporter-2022-09-05T20.parquet"
    other = dict(REAL_PAYLOAD, sc="W1AW", rc="K1ABC", sl="FN31", rl="FN42")
    write_hourly_parquet([REAL_PAYLOAD], hour=dt.datetime(2022, 9, 5, 20, tzinfo=dt.timezone.utc), out_path=out_path)
    write_hourly_parquet([other], hour=dt.datetime(2022, 9, 5, 20, tzinfo=dt.timezone.utc), out_path=out_path)
    spots = pl.read_parquet(out_path)
    assert spots.height == 2
    assert set(spots["dx_call"].to_list()) == {"SP2EWQ", "W1AW"}


class _FakeMqttClient:
    """Stands in for paho.mqtt.client.Client -- no network, records what the
    accumulator would have done to a real client."""

    def __init__(self):
        self.on_connect = None
        self.on_message = None
        self.on_disconnect = None
        self.subscriptions: list[str] = []

    def subscribe(self, topic):
        self.subscriptions.append(topic)


def test_accumulator_subscribes_on_every_connect_including_reconnect():
    fake = _FakeMqttClient()
    PSKReporterAccumulator(out_dir="unused", band="20m", client_factory=lambda: fake)
    assert fake.on_connect is not None
    fake.on_connect(fake, None, {}, 0)
    fake.on_connect(fake, None, {}, 0)  # simulated reconnect
    assert fake.subscriptions == ["pskr/filter/v2/20m/#", "pskr/filter/v2/20m/#"]


def test_accumulator_flushes_on_hour_boundary(tmp_path):
    flushed: list[FlushResult] = []

    def fake_flush(buffer, hour, out_path):
        result = FlushResult(hour=hour, path=out_path, n_received=len(buffer), n_parsed=len(buffer), n_qualifying=len(buffer))
        flushed.append(result)
        return result

    acc = PSKReporterAccumulator(out_dir=tmp_path, client_factory=_FakeMqttClient, on_flush=fake_flush)

    hour0 = dict(REAL_PAYLOAD, t_tx=1662404400)  # 2022-09-05 19:00:00 UTC
    hour1 = dict(REAL_PAYLOAD, t_tx=1662408000)  # 2022-09-05 20:00:00 UTC

    acc.ingest(hour0)
    acc.ingest(hour0)
    assert flushed == []  # still buffering hour 19
    acc.ingest(hour1)  # crossing into hour 20 flushes hour 19's buffer
    assert len(flushed) == 1
    assert flushed[0].n_received == 2

    acc.flush()  # explicit flush (e.g. shutdown) drains what's left
    assert len(flushed) == 2
    assert flushed[1].n_received == 1


def test_accumulator_flush_on_empty_buffer_is_noop(tmp_path):
    acc = PSKReporterAccumulator(out_dir=tmp_path, client_factory=_FakeMqttClient)
    assert acc.flush() is None
