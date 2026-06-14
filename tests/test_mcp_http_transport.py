import http.server
import io
import json
import sys
from typing import Optional

import chromadb  # noqa: F401
import pytest

from mempalace import mcp_server


class _FakeSocket:
    def __init__(self, request_bytes: bytes):
        self._read = io.BytesIO(request_bytes)
        self._written = io.BytesIO()

    def makefile(self, mode, buffering=None):
        if "r" in mode:
            return self._read
        return self._written

    def sendall(self, data: bytes):
        self._written.write(data)

    def close(self):
        pass

    def response_bytes(self) -> bytes:
        return self._written.getvalue()


def _capture_http_handler(monkeypatch):
    captured = {}

    class _FakeHTTPServer:
        daemon_threads = True
        allow_reuse_address = True

        def __init__(self, server_address, handler_cls):
            captured["server_address"] = server_address
            captured["handler_cls"] = handler_cls

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def serve_forever(self, poll_interval=0.5):
            captured["poll_interval"] = poll_interval

    monkeypatch.setattr(http.server, "ThreadingHTTPServer", _FakeHTTPServer)

    mcp_server._serve_http("127.0.0.1", 8765)

    assert captured["server_address"] == ("127.0.0.1", 8765)
    assert captured["poll_interval"] == 0.5
    return captured["handler_cls"]


def _run_raw_request(handler_cls, raw_request: bytes) -> bytes:
    sock = _FakeSocket(raw_request)
    handler_cls(sock, ("127.0.0.1", 12345), object())
    return sock.response_bytes()


def _build_request(
    method: str,
    path: str,
    body: Optional[bytes] = None,
    headers: Optional[dict] = None,
) -> bytes:
    body = body or b""
    headers = dict(headers or {})
    headers.setdefault("Host", "127.0.0.1")
    headers.setdefault("Connection", "close")
    headers.setdefault("Content-Length", str(len(body)))

    head = [f"{method} {path} HTTP/1.1"]
    head.extend(f"{key}: {value}" for key, value in headers.items())
    return ("\r\n".join(head) + "\r\n\r\n").encode("ascii") + body


def _parse_response(raw_response: bytes):
    head, _, body = raw_response.partition(b"\r\n\r\n")
    status_line = head.splitlines()[0].decode("iso-8859-1")
    status = int(status_line.split()[1])
    return status, body


def _fake_dispatch(request):
    method = request.get("method")
    req_id = request.get("id")

    if method == "initialize":
        params = request.get("params") or {}
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": params.get("protocolVersion", "2025-11-25"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mempalace", "version": "test"},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        tools = [
            {
                "name": f"tool_{idx}",
                "description": "test tool",
                "inputSchema": {"type": "object", "properties": {}},
            }
            for idx in range(128)
        ]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

    if method == "notifications/initialized":
        return None

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": "Method not found"},
    }


@pytest.fixture()
def http_handler(monkeypatch):
    monkeypatch.setattr(mcp_server, "handle_request", _fake_dispatch)
    return _capture_http_handler(monkeypatch)


def _rpc(handler_cls, method: str, params: Optional[dict] = None, req_id: int = 1):
    payload = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params or {},
    }
    raw = _run_raw_request(
        handler_cls,
        _build_request(
            "POST",
            "/mcp",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        ),
    )
    status, body = _parse_response(raw)
    return status, json.loads(body.decode("utf-8")) if body else None


def test_parse_args_defaults_to_stdio(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mempalace-mcp"])

    args = mcp_server._parse_args()

    assert args.transport == "stdio"
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_parse_args_accepts_http_transport(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mempalace-mcp",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
        ],
    )

    args = mcp_server._parse_args()

    assert args.transport == "http"
    assert args.host == "0.0.0.0"
    assert args.port == 9999


def test_http_transport_serves_healthz(http_handler):
    raw = _run_raw_request(http_handler, _build_request("GET", "/healthz"))
    status, body = _parse_response(raw)

    assert status == 200
    assert body == b"ok\n"


def test_http_transport_serves_initialize_ping_and_repeated_tools_list(http_handler):
    status, initialized = _rpc(
        http_handler,
        "initialize",
        {"protocolVersion": "2025-11-25"},
        req_id=1,
    )
    assert status == 200
    assert initialized["result"]["protocolVersion"] == "2025-11-25"

    status, ping = _rpc(http_handler, "ping", {}, req_id=2)
    assert status == 200
    assert ping["result"] == {}

    status, first = _rpc(http_handler, "tools/list", {}, req_id=3)
    assert status == 200
    tools = first["result"]["tools"]
    assert len(tools) == 128
    assert all("name" in tool and "inputSchema" in tool for tool in tools)

    # Regression shape for #1801: repeated large tools/list frames should
    # keep succeeding over HTTP without relying on stdio framing.
    for req_id in range(4, 12):
        status, payload = _rpc(http_handler, "tools/list", {}, req_id=req_id)
        assert status == 200
        assert payload["id"] == req_id
        assert payload["result"]["tools"] == tools


def test_http_transport_returns_parse_error_for_invalid_json(http_handler):
    raw = _run_raw_request(
        http_handler,
        _build_request(
            "POST",
            "/mcp",
            body=b"not-json",
            headers={"Content-Type": "application/json"},
        ),
    )
    status, body = _parse_response(raw)
    payload = json.loads(body.decode("utf-8"))

    assert status == 400
    assert payload["error"]["code"] == -32700
    assert payload["error"]["message"] == "Parse error"


def test_http_transport_accepts_notifications_without_body(http_handler):
    payload = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    raw = _run_raw_request(
        http_handler,
        _build_request(
            "POST",
            "/mcp",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        ),
    )
    status, body = _parse_response(raw)

    assert status == 202
    assert body == b""


def test_http_transport_returns_404_for_unknown_path(http_handler):
    raw = _run_raw_request(
        http_handler,
        _build_request(
            "POST",
            "/not-mcp",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        ),
    )
    status, _body = _parse_response(raw)

    assert status == 404
