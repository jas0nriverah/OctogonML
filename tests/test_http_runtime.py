"""Operational boundaries must remain bounded, correlated and secret-free."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest

from cagemetrics.http_runtime import (
    HTTPMetrics,
    HTTPRuntimeMiddleware,
    HTTPSettings,
    LOGGER,
    ResponseStreamError,
    RequestJSONFormatter,
)


@pytest.fixture
def runtime():
    app, metrics = FastAPI(), HTTPMetrics()
    app.state.writes = 0

    @app.get("/items/{identity}")
    def item(identity: str):
        return {"identity": identity}

    @app.post("/echo")
    async def echo(request: Request):
        app.state.writes += 1
        return {"bytes": len(await request.body())}

    @app.get("/explode")
    def explode():
        raise RuntimeError("database-password-and-provider-key-must-stay-private")

    app.add_middleware(HTTPRuntimeMiddleware, settings=HTTPSettings(max_request_bytes=1024), metrics=metrics)
    with TestClient(app) as client:
        yield client, app, metrics


def test_supplied_uuid_is_correlated_and_invalid_ids_are_replaced(runtime):
    client, _, _ = runtime
    identity = "d6e743bf-a1ed-4a5c-a7ea-f67ad32afffe"
    reply = client.get("/items/a", headers={"X-Request-ID": identity})
    assert reply.headers["X-Request-ID"] == identity
    assert reply.headers["Server-Timing"].startswith("app;dur=")
    assert reply.headers["X-Content-Type-Options"] == "nosniff"
    assert reply.headers["X-Frame-Options"] == "DENY"
    for invalid in ("not-a-uuid", "x" * 200, ""):
        response = client.get("/items/a", headers={"X-Request-ID": invalid})
        assert str(UUID(response.headers["X-Request-ID"])) == response.headers["X-Request-ID"]


def test_oversized_request_is_rejected_before_the_write_handler(runtime):
    client, app, metrics = runtime
    response = client.post("/echo", content=b"x" * 1025)
    assert response.status_code == 413 and app.state.writes == 0
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert 'status="413"' in metrics.render()
    assert client.post("/echo", content=b"x" * 1024).json() == {"bytes": 1024}
    assert app.state.writes == 1


@pytest.mark.parametrize("claimed_length", [None, b"1"])
def test_chunked_body_and_false_content_length_cannot_bypass_limit(claimed_length):
    entered, outgoing = [], []

    async def app(scope, receive, send):
        entered.append(True)

    async def scenario():
        messages = iter(
            [
                {"type": "http.request", "body": b"x" * 600, "more_body": True},
                {"type": "http.request", "body": b"y" * 600, "more_body": False},
            ]
        )

        async def receive():
            return next(messages)

        async def send(message):
            outgoing.append(message)

        headers = [] if claimed_length is None else [(b"content-length", claimed_length)]
        middleware = HTTPRuntimeMiddleware(app, settings=HTTPSettings(1024), metrics=HTTPMetrics())
        await middleware({"type": "http", "method": "POST", "headers": headers}, receive, send)

    asyncio.run(scenario())
    assert not entered and outgoing[0]["status"] == 413


def test_valid_chunked_body_is_preserved_exactly():
    bodies, outgoing = [], []

    async def app(scope, receive, send):
        message = await receive()
        bodies.append(message["body"])
        assert message["more_body"] is False
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def scenario():
        messages = iter(
            [
                {"type": "http.request", "body": b"a" * 512, "more_body": True},
                {"type": "http.request", "body": b"b" * 512, "more_body": False},
            ]
        )

        async def receive():
            return next(messages)

        async def send(message):
            outgoing.append(message)

        middleware = HTTPRuntimeMiddleware(app, settings=HTTPSettings(1024), metrics=HTTPMetrics())
        await middleware({"type": "http", "method": "POST", "headers": []}, receive, send)

    asyncio.run(scenario())
    assert bodies == [b"a" * 512 + b"b" * 512]
    assert outgoing[0]["status"] == 200


def test_errors_and_logs_never_disclose_query_body_or_exception_values(runtime, caplog, monkeypatch):
    client, _, _ = runtime
    monkeypatch.setattr(LOGGER, "propagate", False)
    LOGGER.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="cagemetrics.http"):
            response = client.get("/explode?api_key=query-secret")
            client.post("/echo", content=b"body-secret")
            client.get("/items/private-fighter-value")
    finally:
        LOGGER.removeHandler(caplog.handler)
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal server error",
        "request_id": response.headers["X-Request-ID"],
    }
    formatted = [
        json.loads(RequestJSONFormatter().format(record))
        for record in caplog.records
        if record.name == "cagemetrics.http"
    ]
    text = json.dumps(formatted)
    for secret in (
        "query-secret",
        "body-secret",
        "private-fighter-value",
        "database-password-and-provider-key",
    ):
        assert secret not in text + response.text
    assert any(row.get("error_type") == "RuntimeError" for row in formatted)
    assert any(row.get("route") == "/items/{identity}" for row in formatted)


def test_metrics_use_templates_and_do_not_grow_with_requested_ids(runtime):
    client, _, metrics = runtime
    for value in range(20):
        client.get(f"/items/fighter-{value}")
        client.get(f"/unknown-{value}")
    output = metrics.render()
    assert 'route="/items/{identity}",method="GET",status="200"} 20' in output
    assert 'route="unmatched",method="GET",status="404"} 20' in output
    assert "fighter-" not in output and "unknown-" not in output
    assert output.count("octogonml_http_requests_total{") == 2
    assert 'le="+Inf"} 20' in output


def test_metrics_snapshots_preserve_counts_under_concurrent_updates():
    metrics = HTTPMetrics()

    def record(_):
        for _ in range(25):
            metrics.observe("/predict", "POST", 201, 0.01)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record, range(8)))
    output = metrics.render()
    assert 'status="201"} 200' in output
    assert 'le="0.01"} 200' in output
    assert 'le="0.005"} 0' in output
    assert 'status="201"} 2.000000000' in output


@pytest.mark.parametrize("value", ["1023", "16777217", "not-a-limit", "1.5"])
def test_invalid_request_budget_fails_without_echoing_value(monkeypatch, value):
    monkeypatch.setenv("CAGEMETRICS_MAX_REQUEST_BYTES", value)
    with pytest.raises(ValueError, match="CAGEMETRICS_MAX_REQUEST_BYTES") as error:
        HTTPSettings.from_env()
    if value == "not-a-limit":
        assert value not in str(error.value)


def test_invalid_logging_configuration_fails_cleanly(monkeypatch):
    monkeypatch.setenv("CAGEMETRICS_LOG_LEVEL", "private-key-value")
    with pytest.raises(ValueError, match="CAGEMETRICS_LOG_LEVEL") as error:
        HTTPSettings.from_env()
    assert "private-key-value" not in str(error.value)


def test_non_http_lifespan_is_forwarded_without_body_handling():
    passed = []

    async def app(scope, receive, send):
        passed.append(scope["type"])

    async def unused():
        raise AssertionError("Lifespan must not be read by HTTP middleware")

    asyncio.run(
        HTTPRuntimeMiddleware(app, settings=HTTPSettings(), metrics=HTTPMetrics())(
            {"type": "lifespan"}, unused, unused
        )
    )
    assert passed == ["lifespan"]


def test_post_header_failure_is_sanitized_and_reported_as_an_aborted_response():
    import traceback
    from types import SimpleNamespace

    metrics, outgoing = HTTPMetrics(), []

    async def app(scope, receive, send):
        scope["route"] = SimpleNamespace(path="/stream")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("private-stream-database-password")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        outgoing.append(message)

    middleware = HTTPRuntimeMiddleware(app, settings=HTTPSettings(), metrics=metrics)
    with pytest.raises(ResponseStreamError) as error:
        asyncio.run(middleware({"type": "http", "method": "GET", "headers": []}, receive, send))
    assert "private-stream-database-password" not in "".join(traceback.format_exception(error.value))
    assert [message["status"] for message in outgoing] == [200]
    output = metrics.render()
    assert 'route="/stream",method="GET",status="200"} 1' in output
    assert 'octogonml_http_responses_aborted_total{route="/stream",method="GET"} 1' in output


def test_empty_chunks_do_not_grow_request_buffer_memory():
    import tracemalloc

    async def app(scope, receive, send):
        assert (await receive())["body"] == b"x"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def scenario():
        remaining = 100000

        async def receive():
            nonlocal remaining
            remaining -= 1
            return {"type": "http.request", "body": b"" if remaining else b"x", "more_body": bool(remaining)}

        async def send(message):
            pass

        middleware = HTTPRuntimeMiddleware(app, settings=HTTPSettings(1024), metrics=HTTPMetrics())
        await middleware({"type": "http", "method": "POST", "headers": []}, receive, send)

    tracemalloc.start()
    try:
        asyncio.run(scenario())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 512 * 1024
