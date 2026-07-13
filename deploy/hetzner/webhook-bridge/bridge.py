#!/usr/bin/env python3
"""Small fail-closed webhook bridge for Hetzner alert delivery.

The bridge deliberately forwards only a compact allowlisted event summary.  It
does not log request bodies, phone numbers, API keys, or the outbound URL.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MAX_BODY_BYTES = 64 * 1024
MAX_MESSAGE_CHARS = 1500
OUTBOUND_TIMEOUT_SECONDS = 10


def _compact_text(value: Any, limit: int = 240) -> str | None:
    if not isinstance(value, (str, int, float, bool)):
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    return text[:limit]


def event_message(payload: dict[str, Any]) -> str:
    """Render only non-secret operational fields in a deterministic order."""

    fields = (
        ("event", payload.get("type", payload.get("event"))),
        ("symbol", payload.get("symbol")),
        ("action", payload.get("action")),
        ("status", payload.get("status")),
        ("reason", payload.get("reason")),
        ("server", payload.get("serverId")),
    )
    parts = []
    for label, raw in fields:
        value = _compact_text(raw)
        if value is not None:
            parts.append(f"{label}={value}")
    if not parts:
        parts.append("event=trader.notification")
    return ("Trader alert: " + " | ".join(parts))[:MAX_MESSAGE_CHARS]


def deliver(message: str) -> None:
    phone = os.environ.get("ALERT_PHONE", "").strip()
    api_key = os.environ.get("CALLMEBOT_APIKEY", "").strip()
    if not phone or not api_key:
        raise RuntimeError("alert delivery is not configured")
    query = urlencode({"phone": phone, "text": message, "apikey": api_key})
    request = Request(
        "https://api.callmebot.com/whatsapp.php?" + query,
        headers={"User-Agent": "trader-webhook-bridge/1"},
        method="GET",
    )
    with urlopen(request, timeout=OUTBOUND_TIMEOUT_SECONDS) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"alert provider returned HTTP {response.status}")


class Handler(BaseHTTPRequestHandler):
    server_version = "TraderWebhookBridge/1"

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._respond(200, {"ok": True})
        else:
            self._respond(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/webhook":
            self._respond(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._respond(400, {"ok": False, "error": "invalid content length"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._respond(413, {"ok": False, "error": "invalid body size"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._respond(400, {"ok": False, "error": "invalid JSON"})
            return
        if not isinstance(payload, dict):
            self._respond(400, {"ok": False, "error": "JSON object required"})
            return
        try:
            deliver(event_message(payload))
        except RuntimeError as exc:
            self._respond(503, {"ok": False, "error": str(exc)})
            return
        except (HTTPError, URLError, TimeoutError):
            self._respond(502, {"ok": False, "error": "alert provider unavailable"})
            return
        self._respond(202, {"ok": True})

    def log_message(self, format: str, *args: Any) -> None:
        # BaseHTTPRequestHandler's standard line contains no body/query here.
        super().log_message(format, *args)


def main() -> None:
    port = int(os.environ.get("PORT", "8081"))
    if not 1 <= port <= 65535:
        raise SystemExit("PORT must be between 1 and 65535")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
