# -*- coding: utf-8 -*-
"""AI Agent Handler v2 (plugin base) contract tests."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from llm_provider_base import (
    CONTRACT_VERSION,
    LlmProviderHandler,
    ProviderContext,
)

TESTS_DIR = Path(__file__).resolve().parent


def _handler(**kwargs: Any) -> LlmProviderHandler:
    return LlmProviderHandler(
        logging.getLogger("kit"), {"configuration": {}}, **kwargs
    )


def test_contract_version_is_v1() -> None:
    assert CONTRACT_VERSION == "1"
    assert LlmProviderHandler.contract_version == "1"


def test_provider_context_defaults() -> None:
    ctx = ProviderContext()
    assert ctx.http_client is None
    assert ctx.credentials == {}
    assert ctx.emit is None
    assert ctx.is_cancelled is None
    assert ctx.pool_name is None


def test_get_function_raises_clearly() -> None:
    """In the plugin contract tools execute in agent_engine — get_function
    must fail loudly if a legacy tool loop still calls it."""
    handler = _handler()
    with pytest.raises(RuntimeError, match="capability_engine.invokeTool"):
        handler.get_function("anything")


def test_invoke_async_funct_is_noop() -> None:
    handler = _handler()
    # No-op: returns None without side effects ("x" is ignored).
    handler.invoke_async_funct("m", "f", "c", x=1)


def test_credentials_property_reads_ctx() -> None:
    ctx = ProviderContext(credentials={"api_key": "k"})
    assert _handler(ctx=ctx).credentials == {"api_key": "k"}
    assert _handler().credentials == {}


def test_emit_payload_shape() -> None:
    recorded = []
    ctx = ProviderContext(
        emit=lambda c: recorded.append(dict(c)), trace_id="t1"
    )
    handler = _handler(ctx=ctx)
    handler.run = {"run_uuid": "r1", "thread_uuid": "th1"}
    handler.send_data_to_stream(
        index=0, data_format="text", chunk_delta="hey", is_message_end=True
    )
    assert len(recorded) == 1
    chunk = recorded[0]
    for key in (
        "message_group_id",
        "event_type",
        "index",
        "data_format",
        "chunk_delta",
        "is_message_end",
        "thread_uuid",
    ):
        assert key in chunk, f"payload missing {key!r}: {chunk!r}"
    assert chunk["message_group_id"] == "t1-r1"
    assert chunk["event_type"] == "chunk"
    assert chunk["is_message_end"] is True


def test_emit_suffix_extends_group_id() -> None:
    recorded = []
    ctx = ProviderContext(emit=lambda c: recorded.append(dict(c)), trace_id="t")
    handler = _handler(ctx=ctx)
    handler.run = {"run_uuid": "r"}
    handler.send_data_to_stream(index=1, chunk_delta="x", suffix="reasoning")
    assert recorded[0]["message_group_id"] == "t-r-reasoning"


def test_emit_kind_marks_reasoning_chunks() -> None:
    recorded = []
    ctx = ProviderContext(emit=lambda c: recorded.append(dict(c)), trace_id="t")
    handler = _handler(ctx=ctx)
    handler.send_data_to_stream(index=0, chunk_delta="why", suffix="rs#0")
    handler.send_data_to_stream(index=1, chunk_delta="answer")
    assert [c["kind"] for c in recorded] == ["reasoning", "text"]


def test_emit_without_ctx_is_noop() -> None:
    handler = _handler()
    handler.send_data_to_stream(index=0, chunk_delta="x")  # must not raise


def test_cancellation_latches_after_first_probe() -> None:
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return True

    ctx = ProviderContext(is_cancelled=cancel, emit=lambda c: None)
    handler = _handler(ctx=ctx)
    assert handler.is_stream_cancelled() is True
    assert calls["n"] == 1
    # Latched: subsequent probes must not call the host again.
    assert handler.is_stream_cancelled() is True
    assert calls["n"] == 1


def test_new_run_clears_cancellation_latch() -> None:
    ctx = ProviderContext(is_cancelled=lambda: True, emit=lambda c: None)
    handler = _handler(ctx=ctx)
    handler.run = {"run_uuid": "a"}
    assert handler.is_stream_cancelled() is True
    handler.run = {"run_uuid": "b"}
    assert handler._stream_closed is False


def test_try_complete_json() -> None:
    handler = _handler()
    assert handler.try_complete_json('{"a": 1}') == '{"a": 1}'
    assert handler.try_complete_json('{"a": 1') == "}"
    assert handler.try_complete_json("[1, 2") == "]"
    assert handler.try_complete_json('{"a": [1]') == "}"
    assert handler.try_complete_json("not json at all") == ""


def test_process_text_content_buffers_and_emits() -> None:
    recorded = []
    ctx = ProviderContext(emit=lambda c: recorded.append(dict(c)), trace_id="t")
    handler = _handler(ctx=ctx)
    handler.run = {"run_uuid": "r"}
    # "hello world" (11 chars) exceeds the default 10-char buffer → emit.
    index, remaining = handler.process_text_content(0, "hello world", "text")
    assert (index, remaining) == (1, "")
    assert recorded[0]["chunk_delta"] == "hello world"


def test_to_native_messages_default_passthrough() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    assert _handler().to_native_messages(msgs) is msgs


def test_plugin_metadata_defaults() -> None:
    class KitHandler(LlmProviderHandler):
        api_protocol = "KIT"
        display_name = "Kit"
        capabilities = {"tools": True}

    md = KitHandler.plugin_metadata()
    assert md["api_protocol"] == "KIT"
    assert md["display_name"] == "Kit"
    assert md["contract_version"] == "1"
    assert md["fallback_protocol"] is None
    assert md["capabilities"] == {"tools": True}
    assert md["pool_name"] == "httpx_llm"
    # The base package ships no configuration_schema.json → None.
    assert md["config_schema"] is None
    assert KitHandler.load_config_schema() is None


def test_num_of_messages_normalisation_kept() -> None:
    handler = LlmProviderHandler(
        logging.getLogger("kit"), {"configuration": {}, "num_of_messages": "5"}
    )
    assert handler.agent["num_of_messages"] == 5
    empty = LlmProviderHandler(
        logging.getLogger("kit"), {"configuration": {}, "num_of_messages": None}
    )
    assert empty.agent["num_of_messages"] == 0


def test_aws_and_mcp_initializers_are_gone() -> None:
    """v2 removes AWS / MCP / core-engine coupling entirely (plan D5)."""
    for name in (
        "_initialize_aws_services",
        "_initialize_message_invoker",
        "_initialize_mcp_http_clients",
        "_run_list_mcp_http_tools",
        "_run_call_mcp_http_tool",
    ):
        assert not hasattr(LlmProviderHandler, name), name
    assert not hasattr(LlmProviderHandler, "task_queue")


def test_fetch_upstream_models_maps_items_and_errors() -> None:
    from types import SimpleNamespace

    import pytest

    from llm_provider_base import bearer, endpoint_base, fetch_upstream_models

    class _Resp:
        def __init__(self, status, body):
            self.status_code, self._body = status, body

        def json(self):
            return self._body

    calls = []

    class _Client:
        def __init__(self, resp):
            self.resp = resp

        def get(self, url, headers=None, params=None, timeout=None):
            calls.append((url, headers))
            return self.resp

    ctx = SimpleNamespace(
        http_client=_Client(_Resp(200, {"models": [{"name": "models/a"}, {"name": ""}, "x"]})),
        endpoint="https://h/v1/", credentials={"api_key": "k"}, timeout_seconds=5,
    )
    assert endpoint_base(ctx) == "https://h/v1"
    assert bearer(ctx) == {"Authorization": "Bearer k"}
    models = fetch_upstream_models(
        ctx, "https://h/m", items_key="models", id_key="name", strip_prefix="models/"
    )
    assert models == [{"model_id": "a", "name": "models/a"}]
    for status, msg in ((401, "Authentication"), (404, "not found"), (500, "Upstream")):
        ctx.http_client = _Client(_Resp(status, {}))
        with pytest.raises(RuntimeError, match=msg):
            fetch_upstream_models(ctx, "https://h/m")


def test_default_test_connection_uses_list_models() -> None:
    from llm_provider_base import LlmProviderHandler

    class _Ok(LlmProviderHandler):
        @classmethod
        def list_models(cls, ctx):
            return [{"model_id": "m", "name": "m"}]

    class _Bad(LlmProviderHandler):
        @classmethod
        def list_models(cls, ctx):
            raise RuntimeError("Authentication failed (HTTP 401)")

    class _Unsupported(LlmProviderHandler):
        @classmethod
        def list_models(cls, ctx):
            return None

    ok = _Ok.test_connection(object())
    assert ok["success"] is True and ok["http_status"] == 200
    bad = _Bad.test_connection(object())
    assert bad["success"] is False and "401" in bad["error_message"]
    assert _Unsupported.test_connection(object()) is None
    assert LlmProviderHandler.test_connection(object()) is None
