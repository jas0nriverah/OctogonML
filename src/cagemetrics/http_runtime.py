"""Bounded ASGI request handling and process-local operational telemetry."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from threading import Lock
import time
from uuid import UUID, uuid4


LOGGER = logging.getLogger("cagemetrics.http")
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 15, 30, 90)
METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}


@dataclass(frozen=True)
class HTTPSettings:
    max_request_bytes: int = 262144
    log_level: str = "INFO"

    def __post_init__(self):
        if type(self.max_request_bytes) is not int or not 1024 <= self.max_request_bytes <= 16777216:
            raise ValueError("CAGEMETRICS_MAX_REQUEST_BYTES must be an integer from 1024 through 16777216")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("CAGEMETRICS_LOG_LEVEL must be DEBUG, INFO, WARNING, or ERROR")

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        try:
            maximum = int(env.get("CAGEMETRICS_MAX_REQUEST_BYTES", "262144"))
        except (ValueError, TypeError):
            raise ValueError(
                "CAGEMETRICS_MAX_REQUEST_BYTES must be an integer from 1024 through 16777216"
            ) from None
        level = env.get("CAGEMETRICS_LOG_LEVEL", "INFO")
        if not isinstance(level, str):
            raise ValueError("CAGEMETRICS_LOG_LEVEL must be DEBUG, INFO, WARNING, or ERROR")
        return cls(maximum, level.upper())


class RequestJSONFormatter(logging.Formatter):
    """Request logs deliberately omit URLs, query values, bodies and credentials."""

    def format(self, record):
        result = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for key in ("request_id", "method", "route", "status", "duration_ms", "error_type", "outcome"):
            if hasattr(record, key):
                result[key] = getattr(record, key)
        return json.dumps(result, separators=(",", ":"), allow_nan=False)


def configure_http_logging(level):
    LOGGER.setLevel(level)
    if not any(isinstance(handler.formatter, RequestJSONFormatter) for handler in LOGGER.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(RequestJSONFormatter())
        LOGGER.addHandler(handler)
    LOGGER.propagate = False


def _label(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class HTTPMetrics:
    """Cumulative counts/histograms keyed by registered route, never a fighter ID."""

    def __init__(self):
        self._lock = Lock()
        self._series = defaultdict(lambda: [0, 0.0, [0] * len(LATENCY_BUCKETS)])
        self._aborted = defaultdict(int)
        self.started_at = time.time()

    def observe(self, route, method, status, seconds, *, aborted=False):
        with self._lock:
            values = self._series[(route, method, status)]
            values[0] += 1
            values[1] += seconds
            for index, boundary in enumerate(LATENCY_BUCKETS):
                values[2][index] += seconds <= boundary
            if aborted:
                self._aborted[(route, method)] += 1

    def render(self):
        with self._lock:
            rows = [(key, (value[0], value[1], tuple(value[2]))) for key, value in self._series.items()]
            aborted = list(self._aborted.items())
        lines = [
            "# HELP octogonml_http_requests_total ASGI requests by route, method and observed status.",
            "# TYPE octogonml_http_requests_total counter",
        ]
        for (route, method, status), (count, _, _) in sorted(rows):
            labels = f'route="{_label(route)}",method="{_label(method)}",status="{status}"'
            lines.append(f"octogonml_http_requests_total{{{labels}}} {count}")
        lines += [
            "# HELP octogonml_http_request_duration_seconds ASGI request elapsed seconds, including body handling.",
            "# TYPE octogonml_http_request_duration_seconds histogram",
        ]
        for (route, method, status), (count, total, buckets) in sorted(rows):
            labels = f'route="{_label(route)}",method="{_label(method)}",status="{status}"'
            for boundary, bucket_count in zip(LATENCY_BUCKETS, buckets):
                lines.append(
                    f'octogonml_http_request_duration_seconds_bucket{{{labels},le="{boundary:g}"}} {bucket_count}'
                )
            lines += [
                f'octogonml_http_request_duration_seconds_bucket{{{labels},le="+Inf"}} {count}',
                f"octogonml_http_request_duration_seconds_count{{{labels}}} {count}",
                f"octogonml_http_request_duration_seconds_sum{{{labels}}} {total:.9f}",
            ]
        lines += [
            "# HELP octogonml_http_responses_aborted_total Responses interrupted after headers were sent.",
            "# TYPE octogonml_http_responses_aborted_total counter",
        ]
        for (route, method), count in sorted(aborted):
            lines.append(
                f'octogonml_http_responses_aborted_total{{route="{_label(route)}",method="{_label(method)}"}} {count}'
            )
        lines += [
            "# TYPE octogonml_process_start_time_seconds gauge",
            f"octogonml_process_start_time_seconds {self.started_at:.6f}",
        ]
        return "\n".join(lines) + "\n"


class ResponseStreamError(RuntimeError):
    """Sanitized failure after the HTTP status can no longer be replaced."""


class HTTPRuntimeMiddleware:
    """Bound a complete request before application handlers can persist writes."""

    def __init__(self, app, *, settings, metrics, on_finish=None):
        self.app, self.settings, self.metrics, self.on_finish = app, settings, metrics, on_finish

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = [value for name, value in scope.get("headers", []) if name.lower() == b"x-request-id"]
        try:
            request_id = str(UUID(headers[0].decode("ascii"))) if len(headers) == 1 else str(uuid4())
        except (ValueError, UnicodeError):
            request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        start = time.perf_counter()
        status, started, aborted = 500, False, False
        method = scope.get("method", "OTHER")
        method = method if method in METHODS else "OTHER"

        async def decorated_send(message):
            nonlocal status, started
            if message["type"] == "http.response.start":
                status, started = message["status"], True
                controlled = {
                    b"x-request-id",
                    b"server-timing",
                    b"x-content-type-options",
                    b"x-frame-options",
                    b"referrer-policy",
                }
                outgoing = [
                    (key, value) for key, value in message.get("headers", []) if key.lower() not in controlled
                ]
                outgoing += [
                    (b"x-request-id", request_id.encode("ascii")),
                    (b"server-timing", f"app;dur={(time.perf_counter() - start) * 1000:.3f}".encode("ascii")),
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                ]
                message = {**message, "headers": outgoing}
            await send(message)

        async def error_response(code, detail):
            body = json.dumps({"detail": detail, "request_id": request_id}).encode()
            await decorated_send(
                {
                    "type": "http.response.start",
                    "status": code,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"cache-control", b"no-store"),
                    ],
                }
            )
            await decorated_send({"type": "http.response.body", "body": body})

        try:
            payload = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    status = 499
                    return
                if message["type"] != "http.request":
                    continue
                part = message.get("body", b"")
                if len(payload) + len(part) > self.settings.max_request_bytes:
                    await error_response(413, "Request body exceeds the configured size limit")
                    return
                payload.extend(part)
                if not message.get("more_body", False):
                    break
            body, consumed = bytes(payload), False

            async def replay_receive():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay_receive, decorated_send)
        except Exception as exc:
            LOGGER.error(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "route": getattr(scope.get("route"), "path", "unmatched"),
                    "error_type": type(exc).__name__,
                },
            )
            if started:
                aborted = True
                raise ResponseStreamError("Response stream terminated after headers") from None
            await error_response(500, "Internal server error")
        finally:
            seconds = time.perf_counter() - start
            route = getattr(scope.get("route"), "path", "unmatched")
            self.metrics.observe(route, method, status, seconds, aborted=aborted)
            if self.on_finish:
                self.on_finish(status, seconds * 1000)
            LOGGER.info(
                "request_finished",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "route": route,
                    "status": status,
                    "duration_ms": round(seconds * 1000, 3),
                    "outcome": "aborted" if aborted else "finished",
                },
            )
