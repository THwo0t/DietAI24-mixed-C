"""Durable exact-request reuse for the C reproduction adaptation."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path[:0] = [str(V2), str(V2 / "src")]
from provider import (Provider, ProviderStop, CHAT_URL, EMBED_URL, CHAT_MODEL, VISION_MODEL,
                          EMBED_MODEL, fingerprint, atomic_json, stamp)


def load(path):
    from round4.src.artifacts import decode
    return decode(Path(path).read_bytes())


def event(**values):
    print(json.dumps({"utc": stamp(), **values}, allow_nan=False), flush=True)


class Store:
    def __init__(self, directory, frozen, total=314):
        self.directory, self.frozen = Path(directory), frozen
        self.total = total
        self.lock = threading.RLock()
        self.completed = set()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, stage, sid):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", stage):
            raise ValueError("invalid_stage_name")
        return self.directory / stage / (fingerprint(sid) + ".json")

    def get(self, stage, sid, binding):
        path = self.path(stage, sid)
        if not path.exists():
            return None
        if path.is_symlink():
            raise ValueError("stage_symlink_rejected")
        saved = load(path)
        expected = {"freeze_sha256": self.frozen, "input": binding}
        if (saved["binding"] != expected or saved["sid"] != sid
                or saved["value_sha256"] != fingerprint(saved["value"])):
            raise ValueError("saved_stage_changed")
        return deepcopy(saved["value"])

    def put(self, stage, sid, binding, value):
        with self.lock:
            existing = self.get(stage, sid, binding)
            if existing is not None:
                if existing != value:
                    raise ValueError("completed_stage_is_immutable")
                return
            atomic_json(self.path(stage, sid), {"sid": sid,
                "binding": {"freeze_sha256": self.frozen, "input": binding},
                "value": value, "value_sha256": fingerprint(value)})
            if stage == "record":
                self.completed.add(sid)
                count = len(list((self.directory / "record").glob("*.json")))
                atomic_json(self.directory / "progress.json", {"status": "running",
                    "completed": count, "total": self.total, "updated": stamp()})
                if count % 10 == 0 or count == self.total:
                    event(profile=self.directory.name, completed=count, total=self.total)

    def run_stage(self, name, inputs, call, validator):
        saved = self.get("stages", name, inputs)
        if saved is not None:
            value = validator(saved)
            if fingerprint(value) != fingerprint(saved):
                raise ValueError("cached_stage_reparse_changed")
            return value
        value = validator(call())
        self.put("stages", name, inputs, value)
        return deepcopy(value)


class CachedProvider(Provider):
    """Keep historical request namespaces and global cumulative attempt ledgers.

    A process lock around the SAME exact request hash prevents two providers
    from sending it twice. No old cache/attempt file is replaced or reset.
    """
    def __init__(self, profile, *, allow_paid=False, keys=None, transport=None, sleep=time.sleep):
        if profile not in ("N5", "ASA"):
            raise ValueError("unknown_profile")
        self.profile, self.stopped = profile, False
        namespace = {"N5": "mae-v2-development", "ASA": "development"}[profile]
        super().__init__(HERE / "run" / profile / "provider", allow_paid=allow_paid,
            namespace=namespace, timeout=600, keys=keys, transport=transport, sleep=sleep)

    @contextmanager
    def transport_slot(self, cache_id):
        """At most 32 live request sequences across ALL three processes."""
        directory = HERE / "run" / "transport_slots"
        directory.mkdir(parents=True, exist_ok=True)
        offset = int(cache_id[:8], 16) % 32
        while not self.stopped:
            for index in range(32):
                stream = (directory / (str((offset + index) % 32) + ".lock")).open("a")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                    stream.close()
                return
            time.sleep(.2)
        raise ProviderStop("Provider has stopped")

    def _request(self, role, payload, *, route=None):
        if self.stopped:
            raise ProviderStop("Provider has stopped")
        url = CHAT_URL if role == "chat" else EMBED_URL
        cache_id = fingerprint({"url": url, "payload": payload,
            "namespace": self.namespace if role == "chat" else "official-query-v1"})
        directory = V2 / "runtime_cache" / "request_locks" / role
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with (directory / (cache_id + ".lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if self.stopped:
                    raise ProviderStop("Provider has stopped")
                cached = (self.cache / role / (cache_id + ".json")).exists()
                inherited = False
                legacy = V2.parent / "runtime_cache/responses" / role / (cache_id + ".json")
                if self.profile == "ASA" and role == "chat" and not cached and legacy.is_file():
                    saved = load(legacy)
                    if (saved.get("request_hash") != cache_id or saved.get("model") != payload["model"]
                            or saved.get("value_hash") != fingerprint(saved.get("value"))
                            or not isinstance(saved.get("value"), str)):
                        raise ProviderStop("Historical ASA exact cache provenance mismatch")
                    atomic_json(self.cache / role / (cache_id + ".json"), saved)
                    cached, inherited = True, True
                if cached:
                    value = super()._request(role, payload, route=route)
                else:
                    with self.transport_slot(cache_id):
                        value = super()._request(role, payload, route=route)
                with self.lock:
                    with (self.run_dir / "observations.jsonl").open("a") as stream:
                        stream.write(json.dumps({"utc": stamp(), "role": role,
                            "request_hash": cache_id, "cache_present_before": cached,
                            "inherited_legacy_cache": inherited, "model_requested": payload["model"],
                            "status": "returned", "dataset": (route or {}).get("dataset"),
                            "stage": (route or {}).get("stage")}) + "\n")
                return value
        except ProviderStop:
            self.stopped = True
            raise
        except (ValueError, OSError, TypeError, KeyError):
            self.stopped = True
            raise ProviderStop("Provider cache or request state is invalid") from None

    def _log(self, record):
        safe = {k: record.get(k) for k in ("started", "role", "endpoint", "model_requested",
                                         "request_hash", "attempt", "status", "finish_reason", "input_has_images", "dataset", "stage")}
        usage = record.get("usage")
        safe["usage"] = {k: v for k, v in (usage.items() if isinstance(usage, dict) else [])
            if k in ("prompt_tokens", "completion_tokens", "total_tokens") and type(v) is int and v >= 0}
        served = record.get("model_served")
        safe["model_served"] = served if served in (CHAT_MODEL, VISION_MODEL, EMBED_MODEL, None) else "unexpected_model"
        error = record.get("error_type")
        safe["error_type"] = error if error in ("ConnectError", "ReadTimeout", "ConnectTimeout", "ReadError",
            "RemoteProtocolError", "rate_limit_error", "server_error") else ("provider_error" if error else None)
        with self.lock:
            with (self.run_dir / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(safe, allow_nan=False) + "\n")
                stream.flush()


def usage_summary(profile):
    directory = HERE / "run" / profile / "provider"
    totals = {r: {"requests": 0, "errors": 0, "prompt_tokens": 0,
                   "completion_tokens": 0, "total_tokens": 0} for r in ("chat", "embedding")}
    path = directory / "requests.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        record = json.loads(line)
        out = totals[record["role"]]
        out["requests"] += 1
        out["errors"] += record.get("status") != 200
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            out[key] += record.get("usage", {}).get(key, 0)
    return {"profile": profile, "new_transmissions": totals,
        "third_party_cost_usd": None, "cost_notice": "Third-party price unverified; completion tokens include reasoning."}
