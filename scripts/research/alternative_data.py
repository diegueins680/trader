#!/usr/bin/env python3
"""Point-in-time ingestion and feature processing for non-exchange data.

The collector deliberately uses a provider-neutral schema.  Paid and free
providers change, but every observation used by a model must preserve the same
three times:

* ``eventTime``: when the underlying event or measurement occurred;
* ``timestamp``: when the observation became available to the trader; and
* ``ingestedAt``: when this collector recorded it.

Only ``timestamp`` is used for as-of joins.  This makes publication lags and
revisions explicit and prevents a later release from leaking into an earlier
training bar.

Sources are configured as local/HTTP CSV, JSON, or RSS feeds.  Each source
produces one numeric metric; multiple metrics from one provider are represented
by multiple source entries.  Credentials are read from named environment
variables and are never written to the cache or status output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import email.utils
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET


SCHEMA_VERSION = 1
FAMILIES = (
    "microstructure",
    "options_vol",
    "onchain",
    "macro",
    "cot",
    "news",
    "filings",
    "policy",
    "fundamentals",
    "stablecoin",
    "institutional_flows",
    "network",
    "developer",
    "governance",
    "attention",
    "social",
    "prediction_market",
    "real_world",
    "security",
)
FAMILY_ALIASES = {
    "options": "options_vol",
    "optionsvol": "options_vol",
    "option_volatility": "options_vol",
    "on_chain": "onchain",
    "central_bank": "policy",
    "government": "policy",
    "company_fundamentals": "fundamentals",
    "stablecoins": "stablecoin",
    "fund_flows": "institutional_flows",
    "custody_flows": "institutional_flows",
    "miner_validator": "network",
    "network_operations": "network",
    "dev": "developer",
    "token_supply": "governance",
    "search": "attention",
    "web": "attention",
    "prediction": "prediction_market",
    "prediction_markets": "prediction_market",
    "operations": "real_world",
}
CANONICAL_FIELDS = (
    "timestamp",
    "eventTime",
    "source",
    "family",
    "metric",
    "entity",
    "value",
    "unit",
    "revision",
    "ingestedAt",
    "aggregation",
    "transform",
    "polarity",
    "maxAgeMs",
    "minHistory",
)
TIME_FIELD_CANDIDATES = (
    "availableTime",
    "available_time",
    "publishedAt",
    "published_at",
    "timestamp",
    "timestampMs",
    "time",
    "date",
    "datetime",
)
EVENT_TIME_FIELD_CANDIDATES = (
    "eventTime",
    "event_time",
    "effectiveAt",
    "effective_at",
) + TIME_FIELD_CANDIDATES
POSITIVE_WORDS = frozenset(
    {
        "adopt",
        "approval",
        "approve",
        "beat",
        "bullish",
        "gain",
        "growth",
        "launch",
        "profit",
        "recover",
        "secure",
        "surge",
        "upgrade",
    }
)
NEGATIVE_WORDS = frozenset(
    {
        "attack",
        "ban",
        "bearish",
        "breach",
        "decline",
        "depeg",
        "exploit",
        "hack",
        "lawsuit",
        "loss",
        "outage",
        "reject",
        "risk",
    }
)


class ConfigError(ValueError):
    """Raised when a source configuration cannot preserve PIT semantics."""


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    family: str
    adapter: str
    location: str
    metric: str
    records_path: str | None
    event_time_field: str | None
    available_time_field: str | None
    lag_ms: int
    value_field: str | None
    value_mode: str
    text_field: str | None
    entity: str
    entity_field: str | None
    unit: str
    unit_field: str | None
    revision_field: str | None
    aggregation: str
    transform: str
    polarity: float
    max_age_ms: int | None
    min_history: int
    query: Mapping[str, str]
    query_from_env: Mapping[str, str]
    headers_from_env: Mapping[str, str]


@dataclass(frozen=True)
class Observation:
    timestamp: int
    eventTime: int
    source: str
    family: str
    metric: str
    entity: str
    value: float
    unit: str
    revision: str
    ingestedAt: int
    aggregation: str
    transform: str
    polarity: float
    maxAgeMs: int | None
    minHistory: int

    def key(self) -> tuple[Any, ...]:
        return (
            self.source,
            self.family,
            self.metric,
            self.entity,
            self.eventTime,
            self.timestamp,
            self.revision,
        )


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    cache_path: Path
    sources: tuple[SourceSpec, ...]


@dataclass(frozen=True)
class SourceResult:
    source: str
    family: str
    status: str
    rows: int
    error: str | None = None


def _normalize_name(value: str) -> str:
    return "".join(c.lower() for c in value if c.isalnum() or c == "_").strip("_")


def normalize_family(value: str) -> str:
    normalized = _normalize_name(value.strip().replace("-", "_").replace(" ", "_"))
    normalized = FAMILY_ALIASES.get(normalized, normalized)
    if normalized not in FAMILIES:
        raise ConfigError(
            f"unknown family {value!r}; expected one of {', '.join(FAMILIES)}"
        )
    return normalized


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, label)


def _mapping_of_strings(value: Any, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_nonempty_string(key, f"{label} key")] = _nonempty_string(
            item, f"{label}.{key}"
        )
    return result


def _duration_ms(value: Any, label: str, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a non-negative number of seconds")
    if value < 0 or not math.isfinite(float(value)):
        raise ConfigError(f"{label} must be a non-negative finite number")
    return round(float(value) * 1000)


def _parse_source(raw: Any, index: int) -> SourceSpec | None:
    if not isinstance(raw, dict):
        raise ConfigError(f"sources[{index}] must be an object")
    if raw.get("enabled", True) is False:
        return None
    source_id = _nonempty_string(raw.get("id"), f"sources[{index}].id")
    family = normalize_family(_nonempty_string(raw.get("family"), f"{source_id}.family"))
    adapter = _nonempty_string(raw.get("adapter"), f"{source_id}.adapter").lower()
    if adapter not in {"csv", "json", "rss"}:
        raise ConfigError(f"{source_id}.adapter must be csv, json, or rss")
    location = _nonempty_string(raw.get("location"), f"{source_id}.location")
    metric = _nonempty_string(raw.get("metric"), f"{source_id}.metric")
    value_mode = str(raw.get("valueMode", "field")).strip().lower()
    if value_mode not in {"field", "count", "sentiment"}:
        raise ConfigError(f"{source_id}.valueMode must be field, count, or sentiment")
    value_field = _optional_string(raw.get("valueField"), f"{source_id}.valueField")
    text_field = _optional_string(raw.get("textField"), f"{source_id}.textField")
    if value_mode == "field" and value_field is None:
        raise ConfigError(f"{source_id}.valueField is required for valueMode=field")
    if value_mode == "sentiment" and text_field is None:
        raise ConfigError(f"{source_id}.textField is required for valueMode=sentiment")
    aggregation = str(raw.get("aggregation", "last")).strip().lower()
    if aggregation not in {"last", "sum", "mean", "count"}:
        raise ConfigError(f"{source_id}.aggregation must be last, sum, mean, or count")
    transform = str(raw.get("transform", "zscore")).strip().lower()
    if transform not in {"raw", "delta", "pct_change", "zscore"}:
        raise ConfigError(
            f"{source_id}.transform must be raw, delta, pct_change, or zscore"
        )
    polarity = raw.get("polarity", 1.0)
    if isinstance(polarity, bool) or not isinstance(polarity, (int, float)):
        raise ConfigError(f"{source_id}.polarity must be numeric")
    polarity = float(polarity)
    if not math.isfinite(polarity) or polarity == 0:
        raise ConfigError(f"{source_id}.polarity must be finite and non-zero")
    min_history = raw.get("minHistory", 20)
    if isinstance(min_history, bool) or not isinstance(min_history, int) or min_history < 1:
        raise ConfigError(f"{source_id}.minHistory must be a positive integer")
    lag_ms = _duration_ms(raw.get("publicationLagSeconds"), f"{source_id}.publicationLagSeconds", 0)
    max_age_ms = _duration_ms(raw.get("maxAgeSeconds"), f"{source_id}.maxAgeSeconds", None)
    assert lag_ms is not None
    return SourceSpec(
        source_id=source_id,
        family=family,
        adapter=adapter,
        location=location,
        metric=metric,
        records_path=_optional_string(raw.get("recordsPath"), f"{source_id}.recordsPath"),
        event_time_field=_optional_string(raw.get("eventTimeField"), f"{source_id}.eventTimeField"),
        available_time_field=_optional_string(raw.get("availableTimeField"), f"{source_id}.availableTimeField"),
        lag_ms=lag_ms,
        value_field=value_field,
        value_mode=value_mode,
        text_field=text_field,
        entity=str(raw.get("entity", "")).strip(),
        entity_field=_optional_string(raw.get("entityField"), f"{source_id}.entityField"),
        unit=str(raw.get("unit", "")).strip(),
        unit_field=_optional_string(raw.get("unitField"), f"{source_id}.unitField"),
        revision_field=_optional_string(raw.get("revisionField"), f"{source_id}.revisionField"),
        aggregation=aggregation,
        transform=transform,
        polarity=polarity,
        max_age_ms=max_age_ms,
        min_history=min_history,
        query=_mapping_of_strings(raw.get("query"), f"{source_id}.query"),
        query_from_env=_mapping_of_strings(
            raw.get("queryFromEnv"), f"{source_id}.queryFromEnv"
        ),
        headers_from_env=_mapping_of_strings(
            raw.get("headersFromEnv"), f"{source_id}.headersFromEnv"
        ),
    )


def load_config(path: str | os.PathLike[str]) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be an object")
    if raw.get("schemaVersion") != SCHEMA_VERSION:
        raise ConfigError(f"schemaVersion must be {SCHEMA_VERSION}")
    cache_raw = raw.get("cache", "../../data/research/alternative-observations.csv")
    cache_text = _nonempty_string(cache_raw, "cache")
    cache_path = Path(cache_text).expanduser()
    if not cache_path.is_absolute():
        cache_path = (config_path.parent / cache_path).resolve()
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list):
        raise ConfigError("sources must be an array")
    sources = tuple(
        source
        for index, value in enumerate(sources_raw)
        if (source := _parse_source(value, index)) is not None
    )
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ConfigError("enabled source ids must be unique")
    return PipelineConfig(config_path, cache_path, sources)


def _resolved_location(config: PipelineConfig, source: SourceSpec) -> str:
    if urllib.parse.urlsplit(source.location).scheme in {"http", "https"}:
        return source.location
    path = Path(source.location).expanduser()
    if not path.is_absolute():
        path = config.path.parent / path
    return str(path.resolve())


def _request_for(source: SourceSpec, location: str) -> urllib.request.Request:
    query = dict(source.query)
    for parameter, env_name in source.query_from_env.items():
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise ConfigError(
                f"{source.source_id} requires environment variable {env_name}"
            )
        query[parameter] = value
    parsed = urllib.parse.urlsplit(location)
    existing = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_string = urllib.parse.urlencode(existing + list(query.items()))
    url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, query_string, parsed.fragment)
    )
    headers = {"User-Agent": "trader-alternative-data/1.0"}
    for header, env_name in source.headers_from_env.items():
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise ConfigError(
                f"{source.source_id} requires environment variable {env_name}"
            )
        headers[header] = value
    return urllib.request.Request(url, headers=headers)


@contextmanager
def _open_source(config: PipelineConfig, source: SourceSpec) -> Iterator[Any]:
    location = _resolved_location(config, source)
    if urllib.parse.urlsplit(location).scheme in {"http", "https"}:
        with urllib.request.urlopen(_request_for(source, location), timeout=30) as response:
            yield response
    else:
        with open(location, "rb") as handle:
            yield handle


def _lookup(value: Any, path: str | None) -> Any:
    if path is None or path == "":
        return value
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _json_records(config: PipelineConfig, source: SourceSpec) -> list[Mapping[str, Any]]:
    with _open_source(config, source) as handle:
        value = json.load(handle)
    selected = _lookup(value, source.records_path)
    if isinstance(selected, Mapping):
        return [selected]
    if isinstance(selected, list):
        return [row for row in selected if isinstance(row, Mapping)]
    raise ValueError(f"{source.source_id} recordsPath does not select an object or array")


def _csv_records(config: PipelineConfig, source: SourceSpec) -> list[Mapping[str, Any]]:
    with _open_source(config, source) as binary:
        rows = binary.read().decode("utf-8-sig").splitlines()
    return list(csv.DictReader(rows))


def _rss_records(config: PipelineConfig, source: SourceSpec) -> list[Mapping[str, Any]]:
    with _open_source(config, source) as handle:
        root = ET.parse(handle).getroot()
    records: list[Mapping[str, Any]] = []
    for item in list(root.findall(".//item")) + list(root.findall(".//{*}entry")):
        record: dict[str, Any] = {}
        for child in item:
            name = child.tag.rsplit("}", 1)[-1]
            if child.text and child.text.strip():
                record[name] = child.text.strip()
            for attr, value in child.attrib.items():
                record[f"{name}.{attr}"] = value
        records.append(record)
    return records


def _timestamp_ms(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a timestamp")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("timestamp is not finite")
        magnitude = abs(number)
        if magnitude < 100_000_000_000:
            number *= 1000
        elif magnitude > 100_000_000_000_000:
            number /= 1000
        return round(number)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp is empty")
    text = value.strip()
    try:
        return _timestamp_ms(float(text))
    except ValueError:
        pass
    try:
        parsed_email = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        parsed_email = None
    if parsed_email is not None:
        if parsed_email.tzinfo is None:
            parsed_email = parsed_email.replace(tzinfo=timezone.utc)
        return round(parsed_email.timestamp() * 1000)
    iso = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = datetime.fromisoformat(iso)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return round(parsed.timestamp() * 1000)


def _first_present(row: Mapping[str, Any], fields: Iterable[str]) -> Any:
    for field in fields:
        value = _lookup(row, field)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("metric value is not finite")
    return number


def _sentiment(value: Any) -> float:
    words = [
        "".join(character for character in token.lower() if character.isalpha())
        for token in str(value).split()
    ]
    words = [word for word in words if word]
    if not words:
        return 0.0
    score = sum(word in POSITIVE_WORDS for word in words) - sum(
        word in NEGATIVE_WORDS for word in words
    )
    return score / math.sqrt(len(words))


def _row_observation(source: SourceSpec, row: Mapping[str, Any], ingested_at: int) -> Observation:
    event_raw = (
        _lookup(row, source.event_time_field)
        if source.event_time_field
        else _first_present(row, EVENT_TIME_FIELD_CANDIDATES)
    )
    if event_raw is None:
        raise ValueError("event timestamp is missing")
    event_time = _timestamp_ms(event_raw)
    available_raw = (
        _lookup(row, source.available_time_field)
        if source.available_time_field
        else _first_present(row, TIME_FIELD_CANDIDATES)
    )
    available_time = (
        _timestamp_ms(available_raw) if available_raw is not None else event_time + source.lag_ms
    )
    if source.available_time_field is None:
        available_time = max(available_time, event_time + source.lag_ms)
    if available_time < event_time:
        raise ValueError("available timestamp precedes event timestamp")
    if source.value_mode == "count":
        value = 1.0
    elif source.value_mode == "sentiment":
        value = _sentiment(_lookup(row, source.text_field))
    else:
        value = _number(_lookup(row, source.value_field))
    entity_raw = _lookup(row, source.entity_field) if source.entity_field else source.entity
    unit_raw = _lookup(row, source.unit_field) if source.unit_field else source.unit
    revision_raw = _lookup(row, source.revision_field) if source.revision_field else ""
    return Observation(
        timestamp=available_time,
        eventTime=event_time,
        source=source.source_id,
        family=source.family,
        metric=source.metric,
        entity="" if entity_raw is None else str(entity_raw).strip(),
        value=value,
        unit="" if unit_raw is None else str(unit_raw).strip(),
        revision="" if revision_raw is None else str(revision_raw).strip(),
        ingestedAt=ingested_at,
        aggregation=source.aggregation,
        transform=source.transform,
        polarity=source.polarity,
        maxAgeMs=source.max_age_ms,
        minHistory=source.min_history,
    )


def fetch_source(config: PipelineConfig, source: SourceSpec, *, now_ms: int | None = None) -> list[Observation]:
    ingested_at = round(time.time() * 1000) if now_ms is None else now_ms
    loaders = {"csv": _csv_records, "json": _json_records, "rss": _rss_records}
    records = loaders[source.adapter](config, source)
    observations: list[Observation] = []
    errors = 0
    for row in records:
        try:
            observations.append(_row_observation(source, row, ingested_at))
        except (TypeError, ValueError):
            errors += 1
    if records and not observations:
        raise ValueError(f"all {len(records)} records were invalid")
    if errors:
        print(
            f"WARN: {source.source_id} ignored {errors} malformed records",
            file=sys.stderr,
        )
    return observations


def _observation_from_row(row: Mapping[str, str]) -> Observation:
    max_age_raw = (row.get("maxAgeMs") or "").strip()
    return Observation(
        timestamp=int(row["timestamp"]),
        eventTime=int(row["eventTime"]),
        source=row["source"],
        family=normalize_family(row["family"]),
        metric=row["metric"],
        entity=row.get("entity", ""),
        value=_number(row["value"]),
        unit=row.get("unit", ""),
        revision=row.get("revision", ""),
        ingestedAt=int(row["ingestedAt"]),
        aggregation=row.get("aggregation", "last"),
        transform=row.get("transform", "zscore"),
        polarity=float(row.get("polarity", "1")),
        maxAgeMs=int(max_age_raw) if max_age_raw else None,
        minHistory=int(row.get("minHistory", "20")),
    )


def read_cache(path: str | os.PathLike[str]) -> list[Observation]:
    cache_path = Path(path)
    if not cache_path.exists():
        return []
    with cache_path.open(newline="", encoding="utf-8") as handle:
        return [_observation_from_row(row) for row in csv.DictReader(handle)]


def _atomic_write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def _cache_lock(cache_path: Path) -> Iterator[None]:
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def merge_cache(path: str | os.PathLike[str], incoming: Iterable[Observation]) -> int:
    cache_path = Path(path)
    with _cache_lock(cache_path):
        merged = {observation.key(): observation for observation in read_cache(cache_path)}
        for observation in incoming:
            previous = merged.get(observation.key())
            if previous is None or observation.ingestedAt >= previous.ingestedAt:
                merged[observation.key()] = observation
        ordered = sorted(
            merged.values(),
            key=lambda row: (row.timestamp, row.source, row.metric, row.entity, row.eventTime),
        )
        _atomic_write_csv(cache_path, CANONICAL_FIELDS, (asdict(row) for row in ordered))
    return len(ordered)


def collect(config: PipelineConfig) -> tuple[list[SourceResult], int]:
    incoming: list[Observation] = []
    results: list[SourceResult] = []
    for source in config.sources:
        try:
            rows = fetch_source(config, source)
            incoming.extend(rows)
            results.append(SourceResult(source.source_id, source.family, "ok", len(rows)))
        except Exception as error:  # isolate provider/schema/network failures
            message = _safe_error(error)
            results.append(SourceResult(source.source_id, source.family, "error", 0, message))
    total = merge_cache(config.cache_path, incoming) if incoming else len(read_cache(config.cache_path))
    return results, total


def _safe_error(error: Exception) -> str:
    """Describe a source failure without echoing request URLs or credentials."""
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code} {error.reason}"[:240]
    if isinstance(error, urllib.error.URLError):
        return f"network error: {error.reason}"[:240]
    if isinstance(error, FileNotFoundError):
        return f"file not found: {error.filename}"[:240]
    if isinstance(error, (ConfigError, ValueError, json.JSONDecodeError)):
        return str(error).replace("\n", " ")[:240]
    return type(error).__name__


def _bar_open_times(path: str | os.PathLike[str]) -> list[int]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("bar CSV has no header")
        normalized = {_normalize_name(name): name for name in reader.fieldnames}
        selected = next(
            (
                normalized[name]
                for name in ("opentime", "opentimems", "open_time", "timestamp")
                if name in normalized
            ),
            None,
        )
        if selected is None:
            raise ValueError("bar CSV needs openTime/openTimeMs/timestamp")
        values = [_timestamp_ms(row[selected]) for row in reader]
    if not values:
        raise ValueError("bar CSV is empty")
    if values != sorted(set(values)):
        raise ValueError("bar open times must be sorted and unique")
    return values


def _infer_interval_ms(open_times: Sequence[int]) -> int:
    if len(open_times) < 2:
        raise ValueError("interval-ms is required for a single bar")
    deltas = [b - a for a, b in zip(open_times, open_times[1:])]
    interval = round(statistics.median(deltas))
    if interval <= 0:
        raise ValueError("bar interval must be positive")
    return interval


def _winsor(value: float, limit: float = 5.0) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-limit, min(limit, value))


def _transform_values(values: Sequence[float | None], mode: str, polarity: float, min_history: int) -> list[float | None]:
    transformed: list[float | None] = []
    previous: float | None = None
    history: list[float] = []
    for raw in values:
        if raw is None:
            transformed.append(None)
            continue
        if mode == "raw":
            signal = raw
        elif mode == "delta":
            signal = 0.0 if previous is None else raw - previous
        elif mode == "pct_change":
            signal = (
                0.0
                if previous is None or abs(previous) <= 1e-12
                else (raw - previous) / abs(previous)
            )
        else:
            if len(history) < min_history:
                signal = 0.0
            else:
                mean = statistics.fmean(history)
                deviation = statistics.stdev(history) if len(history) > 1 else 0.0
                signal = 0.0 if deviation <= 1e-12 else (raw - mean) / deviation
        transformed.append(_winsor(polarity * signal))
        previous = raw
        history.append(raw)
    return transformed


def _metric_values_for_bars(
    rows: Sequence[Observation],
    bar_closes: Sequence[int],
    bar_opens: Sequence[int] | None = None,
) -> tuple[list[float | None], list[bool]]:
    aggregation = rows[0].aggregation
    if any(
        (row.aggregation, row.transform, row.polarity, row.maxAgeMs, row.minHistory)
        != (
            rows[0].aggregation,
            rows[0].transform,
            rows[0].polarity,
            rows[0].maxAgeMs,
            rows[0].minHistory,
        )
        for row in rows[1:]
    ):
        raise ValueError(f"processing settings changed within metric {rows[0].source}/{rows[0].metric}")
    if aggregation == "last":
        update_signals = _transform_values(
            [row.value for row in rows],
            rows[0].transform,
            rows[0].polarity,
            rows[0].minHistory,
        )
        values: list[float | None] = []
        present: list[bool] = []
        position = 0
        current: Observation | None = None
        current_signal: float | None = None
        for close in bar_closes:
            while position < len(rows) and rows[position].timestamp <= close:
                current = rows[position]
                current_signal = update_signals[position]
                position += 1
            stale = (
                current is None
                or (current.maxAgeMs is not None and close - current.timestamp > current.maxAgeMs)
            )
            values.append(None if stale else current_signal)
            present.append(not stale)
        return values, present

    if bar_opens is not None and len(bar_opens) != len(bar_closes):
        raise ValueError("bar open/close grids have different lengths")
    raw: list[float | None] = []
    present = []
    position = 0
    for index, close in enumerate(bar_closes):
        if bar_opens is not None:
            lower_bound = bar_opens[index]
        elif index > 0:
            lower_bound = bar_closes[index - 1] + 1
        elif len(bar_closes) > 1:
            lower_bound = close - (bar_closes[1] - close) + 1
        else:
            lower_bound = -10**30
        bucket: list[Observation] = []
        while position < len(rows) and rows[position].timestamp <= close:
            if rows[position].timestamp >= lower_bound:
                bucket.append(rows[position])
            position += 1
        values = [row.value for row in bucket]
        if aggregation == "count":
            raw.append(float(len(values)))
        elif aggregation == "sum":
            raw.append(sum(values))
        else:
            raw.append(statistics.fmean(values) if values else 0.0)
        present.append(bool(values))
    transformed = _transform_values(
        raw, rows[0].transform, rows[0].polarity, rows[0].minHistory
    )
    return transformed, present


def build_panel(
    cache_path: str | os.PathLike[str],
    bars_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    interval_ms: int | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    open_times = _bar_open_times(bars_path)
    interval = _infer_interval_ms(open_times) if interval_ms is None else interval_ms
    if interval <= 0:
        raise ValueError("interval-ms must be positive")
    bar_closes = [value + interval - 1 for value in open_times]
    symbol_clean = "" if symbol is None else symbol.strip().upper()
    if symbol is not None and not symbol_clean:
        raise ValueError("symbol must be non-empty when supplied")
    base_symbol = symbol_clean
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if base_symbol.endswith(quote) and len(base_symbol) > len(quote):
            base_symbol = base_symbol[: -len(quote)]
            break
    observations = [
        row
        for row in read_cache(cache_path)
        if row.timestamp <= bar_closes[-1]
        and (
            not symbol_clean
            or not row.entity
            or row.entity.upper() in {symbol_clean, base_symbol}
        )
    ]
    grouped: dict[tuple[str, str, str, str], list[Observation]] = defaultdict(list)
    for row in observations:
        grouped[(row.source, row.family, row.metric, row.entity)].append(row)
    family_series: dict[str, list[list[float | None]]] = defaultdict(list)
    family_presence: dict[str, list[list[bool]]] = defaultdict(list)
    for key, rows in grouped.items():
        rows.sort(key=lambda row: (row.timestamp, row.ingestedAt, row.eventTime))
        values, present = _metric_values_for_bars(rows, bar_closes, open_times)
        family_series[key[1]].append(values)
        family_presence[key[1]].append(present)
    panel_rows: list[dict[str, Any]] = []
    for index, close in enumerate(bar_closes):
        row: dict[str, Any] = {"timestamp": close, "symbol": symbol_clean}
        for family in FAMILIES:
            metric_values = [
                series[index]
                for series in family_series.get(family, [])
                if series[index] is not None
            ]
            row[family] = (
                _winsor(statistics.fmean(metric_values)) if metric_values else 0.0
            )
            presence = family_presence.get(family, [])
            row[f"{family}_coverage"] = (
                sum(series[index] for series in presence) / len(presence) if presence else 0.0
            )
        panel_rows.append(row)
    fields = ["timestamp", "symbol"] + [
        item for family in FAMILIES for item in (family, f"{family}_coverage")
    ]
    output = Path(output_path)
    _atomic_write_csv(output, fields, panel_rows)
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cache": str(Path(cache_path).resolve()),
        "bars": str(Path(bars_path).resolve()),
        "output": str(output.resolve()),
        "intervalMs": interval,
        "barsCount": len(open_times),
        "observationsCount": len(observations),
        "symbol": symbol_clean or None,
        "metricsByFamily": {
            family: len(family_series.get(family, [])) for family in FAMILIES
        },
        "pointInTime": True,
    }
    if manifest_path is not None:
        manifest_output = Path(manifest_path)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=manifest_output.parent,
            prefix=f".{manifest_output.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest_output)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


def _status_json(results: Sequence[SourceResult], total: int, cache_path: Path) -> dict[str, Any]:
    failed = sum(result.status != "ok" for result in results)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "state": "ok" if failed == 0 else "degraded",
        "cache": str(cache_path),
        "cacheRows": total,
        "sources": [asdict(result) for result in results],
        "failedSources": failed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and point-in-time align non-exchange trading data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect", help="refresh the canonical cache")
    collect_parser.add_argument("--config", required=True)
    collect_parser.add_argument("--allow-partial", action="store_true")
    panel_parser = subparsers.add_parser("panel", help="build a bar-aligned model feature CSV")
    panel_parser.add_argument("--cache", required=True)
    panel_parser.add_argument("--bars", required=True)
    panel_parser.add_argument("--output", required=True)
    panel_parser.add_argument("--interval-ms", type=int)
    panel_parser.add_argument("--manifest")
    panel_scope = panel_parser.add_mutually_exclusive_group(required=True)
    panel_scope.add_argument("--symbol")
    panel_scope.add_argument("--global", dest="global_panel", action="store_true")
    run_parser = subparsers.add_parser("run", help="collect and build a feature panel")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--bars", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--interval-ms", type=int)
    run_parser.add_argument("--manifest")
    run_scope = run_parser.add_mutually_exclusive_group(required=True)
    run_scope.add_argument("--symbol")
    run_scope.add_argument("--global", dest="global_panel", action="store_true")
    run_parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "panel":
            manifest = build_panel(
                args.cache,
                args.bars,
                args.output,
                interval_ms=args.interval_ms,
                manifest_path=args.manifest,
                symbol=args.symbol,
            )
            print(json.dumps(manifest, sort_keys=True))
            return 0
        config = load_config(args.config)
        results, total = collect(config)
        status = _status_json(results, total, config.cache_path)
        if args.command == "run":
            status["panel"] = build_panel(
                config.cache_path,
                args.bars,
                args.output,
                interval_ms=args.interval_ms,
                manifest_path=args.manifest,
                symbol=args.symbol,
            )
        print(json.dumps(status, sort_keys=True))
        return 0 if args.allow_partial or status["failedSources"] == 0 else 2
    except (ConfigError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"alternative-data: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
