#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI Agent Handler — provider plugin base (v2, BanyanOS llm_engine contract v1).

``v2.0.0`` — **breaking change** (LLM Provider Plugin Integration plan D5):

- The multi-turn tool loop is **owned by agent_engine**; a handler performs
  exactly one model call and returns either the final answer or the requested
  tool calls (``tool_mode="return"``). Tools execute through
  ``capability_engine.invokeTool``, never inside the handler.
- Outward effects are host-injected through a ``ProviderContext`` supplied
  per call by ``llm_engine``'s ``HandlerProviderAdapter``:

  - ``send_data_to_stream`` → ``ctx.emit(chunk)`` (chunk buffering kept);
  - ``is_stream_cancelled`` → ``ctx.is_cancelled()``;
  - credentials come from ``ctx.credentials`` — never from
    ``agent["configuration"]``;
  - the HTTP client is the pooled ``httpx.Client`` on ``ctx.http_client``
    (SilvaEngine Iron Rule #5: handlers must not build HTTP clients).

- Removed (no longer imported at module level): ``boto3`` /
  ``botocore``, ``mcp_http_client``, ``silvaengine_utility.invoker.Invoker``
  and the eager ``ai_agent_core_engine`` message invoker. Old SilvaEngine
  consumers must pin ``AI-Agent-Handler<2``.

Legacy no-op behaviour: ``invoke_async_funct`` is a no-op and ``get_function``
raises — tool execution is agent_engine's responsibility.
"""

from __future__ import annotations

__author__ = "bibow"

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional

#: Plugin contract version this base implements (plan §5.1).
CONTRACT_VERSION = "1"


@dataclass
class ProviderContext:
    """Host-injected per-call context (built by llm_engine's adapter).

    One instance is created per model call; nothing here is shared between
    concurrent calls. Fields per LLM_PROVIDER_PLUGIN_DEVELOPMENT_PLAN §5.2:

    - ``http_client``: pooled ``httpx.Client`` borrowed from the provider's
      connection pool for the whole call (including a multi-minute stream).
    - ``credentials``: decrypted per-model credentials (e.g.
      ``{"api_key": "..."}``); never logged, never placed on pool headers.
    - ``endpoint``: the model's configured base endpoint (SDKs build their
      own absolute URLs from it).
    - ``model_identifier``: upstream model identifier for this call.
    - ``options``: validated per-model options (against ``config_schema``).
    - ``emit``: stream chunk callback (Redis ``XADD`` on the llm_engine side).
    - ``is_cancelled``: cooperative cancellation check (Redis cancel key).
    """

    http_client: Any = None
    credentials: Dict[str, Any] = field(default_factory=dict)
    endpoint: Optional[str] = None
    model_identifier: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    trace_id: Optional[str] = None
    logger: Optional[logging.Logger] = None
    emit: Optional[Callable[[Dict[str, Any]], None]] = None
    is_cancelled: Optional[Callable[[], bool]] = None
    pool_name: Optional[str] = None


class LlmProviderHandler:
    """Base class for LLM provider handler plugins.

    Subclasses declare plugin metadata as class attributes (plan §5.1) and
    implement ``ask_model``. The BanyanOS host instantiates the handler per
    call — do not rely on cached instances or cross-call state.
    """

    # ------------------------------------------------------------------
    # Plugin metadata (each provider handler overrides)
    # ------------------------------------------------------------------
    #: Unique protocol id registered in llm_engine's ProviderRegistry.
    api_protocol: Optional[str] = None
    #: UI label for the provider.
    display_name: str = ""
    #: Plugin contract version the handler implements.
    contract_version: str = CONTRACT_VERSION
    #: Explicit fallback protocol, or None. The registry never silently
    #: falls back (plan Q9); this is surfaced metadata only.
    fallback_protocol: Optional[str] = None
    #: llm_engine built-in protocol id this plugin takes over (plan D8),
    #: e.g. ``"ANTHROPIC"``. Model records stored with that id resolve to
    #: this plugin when ``provider_impl=plugin`` and after the built-ins are
    #: removed. ``None`` for brand-new providers.
    replaces_protocol: Optional[str] = None
    #: Capability flags: streaming / tools / vision / files / reasoning /
    #: prompt_cache / provider_tools / list_models / citations.
    capabilities: Dict[str, Any] = {}
    #: Default connection pool for this provider; llm_engine falls back to
    #: ``httpx_llm`` when the declared pool is not registered.
    default_pool_name: str = "httpx_llm"
    #: Name of the JSON-schema file shipped next to the handler module.
    config_schema_file: str = "configuration_schema.json"
    #: Role used for tool result messages in the neutral format.
    tool_call_role: str = "tool"

    def __init__(
        self,
        logger: logging.Logger,
        agent: Dict[str, Any],
        ctx: Optional[ProviderContext] = None,
        tool_mode: str = "return",
        **setting: Any,
    ) -> None:
        """Initialize the handler.

        Args:
            logger: Logger instance for recording events and errors.
            agent: Agent dict assembled by llm_engine's adapter
                (``instructions``, ``configuration`` with model options and
                tools, ``llm.llm_name``, ``tool_call_role``).
            ctx: Host-injected ProviderContext (per call).
            tool_mode: ``"return"`` (BanyanOS default) — the handler returns
                tool calls instead of executing them.
            setting: Extra kwargs (kept for signature compatibility).
        """
        self.setting = setting
        self.logger = logger
        self.agent = agent
        self.ctx = ctx
        self.tool_mode = tool_mode

        self._run: Optional[Dict[str, Any]] = None
        # Set True once the host signals cancellation (ctx.is_cancelled) so
        # streaming loops stop generating early.
        self._stream_closed = False

        if "num_of_messages" in self.agent:
            raw: Any = self.agent.get("num_of_messages")
            self.agent["num_of_messages"] = int(raw) if raw else 0

        # Will hold partial text from streaming
        self.accumulated_text: str = ""
        # Will hold the final output message data, if any
        self.final_output: Dict[str, Any] = {}
        self.uploaded_files: List[Dict[str, Any]] = []
        # Kept for subclass compatibility; the agent_engine-owned loop no
        # longer appends here.
        self._short_term_memory: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Run / credentials accessors
    # ------------------------------------------------------------------
    @property
    def run(self) -> Optional[Dict[str, Any]]:
        """Run identity holder (``run_uuid`` / ``thread_uuid``), used only to
        build ``message_group_id`` for stream chunks."""
        return self._run

    @run.setter
    def run(self, value: Optional[Dict[str, Any]]) -> None:
        self._run = value
        # A new run starts connected — clear any cancellation latched by a
        # previous run on a reused handler instance.
        self._stream_closed = False

    @property
    def credentials(self) -> Dict[str, Any]:
        """Per-model credentials injected by the host (decrypted by
        llm_engine's ProviderCaller). Subclasses must read keys from here —
        never from ``agent["configuration"]`` (plan CREDS)."""
        if self.ctx is not None and getattr(self.ctx, "credentials", None):
            return self.ctx.credentials
        return {}

    # ------------------------------------------------------------------
    # Plugin metadata helpers
    # ------------------------------------------------------------------
    @classmethod
    def load_config_schema(cls) -> Optional[Dict[str, Any]]:
        """Load the ``configuration_schema.json`` shipped with the subclass
        package (next to the module file), or None when absent/invalid."""
        try:
            import inspect

            module_path = Path(inspect.getfile(cls)).parent
            schema_path = module_path / cls.config_schema_file
            if not schema_path.is_file():
                return None
            return json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — metadata loading must never break
            return None

    @classmethod
    def plugin_metadata(cls) -> Dict[str, Any]:
        """Metadata dict surfaced through llm_engine's ``llmProviders``
        GraphQL query (plan §6.3): apiProtocol, displayName, contractVersion,
        fallbackProtocol, capabilities, configSchema, poolName."""
        return {
            "api_protocol": cls.api_protocol,
            "display_name": cls.display_name or cls.__name__,
            "contract_version": cls.contract_version,
            "fallback_protocol": cls.fallback_protocol,
            "replaces_protocol": cls.replaces_protocol,
            "capabilities": dict(cls.capabilities or {}),
            "config_schema": cls.load_config_schema(),
            "pool_name": cls.default_pool_name,
        }

    # ------------------------------------------------------------------
    # Model call (subclass contract)
    # ------------------------------------------------------------------
    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        input_files: List[str] = None,
        model_setting: Dict[str, Any] = None,
    ):
        """Perform exactly one model call (single-turn, tool_mode="return").

        Subclasses must implement this and honour ``tool_mode="return"``:
        populate ``final_output.tool_calls`` / ``provider_state`` /
        ``finish_reason`` / ``usage`` and return instead of executing tools.
        """
        raise NotImplementedError("Subclasses must implement the handle method.")

    def to_native_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert neutral messages (+ ``provider_state``) to the provider's
        native format. Default: pass through (neutral == native for
        OpenAI-style protocols). Providers with a different wire format
        override this and reuse their ``_update_conversation_history`` logic.

        Neutral format (plan §5.5): OpenAI-style messages — ``role`` /
        ``content`` / optional ``tool_calls`` / ``tool_call_id`` / optional
        ``content_blocks`` and ``provider_state`` on assistant messages.
        """
        return messages

    # ------------------------------------------------------------------
    # Host hooks (replacements for core-engine / gateway coupling)
    # ------------------------------------------------------------------
    def invoke_async_funct(
        self,
        module_name: str,
        function_name: str,
        class_name: str,
        **params: Any,
    ) -> None:
        """No-op in the plugin contract.

        In ``tool_mode="return"`` tool calls are recorded by agent_engine;
        the handler must not invoke core-engine async functions. Kept for
        signature compatibility with not-yet-migrated subclasses.
        """
        return

    def get_function(self, function_name: str) -> Callable:
        """Not reachable in ``tool_mode="return"`` — tool execution is owned
        by agent_engine (capability_engine.invokeTool). Fails loudly if a
        legacy tool loop still calls it."""
        raise RuntimeError(
            "get_function() is not available in the BanyanOS plugin contract: "
            "tools execute in agent_engine via capability_engine.invokeTool. "
            f"(requested function: {function_name!r})"
        )

    def send_data_to_stream(
        self,
        index: int = 0,
        data_format: str = "text",
        chunk_delta: str = "",
        is_message_end: bool = False,
        suffix: str = "",
    ) -> None:
        """Emit one chunk through the host (``ctx.emit`` → Redis Streams).

        Payload shape follows plan §7.2 (client event schema) so the
        agent_engine reader can forward it to the WebSocket unchanged.
        Without a context/emit (legacy non-plugin deployments) this is a
        logged no-op — there is nowhere to send.
        """
        if self.ctx is None or getattr(self.ctx, "emit", None) is None:
            return

        # Cooperative cancellation: once the host signals cancellation there
        # is nowhere to send, so skip the emit entirely.
        if self.is_stream_cancelled():
            return

        run_uuid = "run"
        if isinstance(self._run, dict) and self._run.get("run_uuid"):
            run_uuid = str(self._run["run_uuid"])
        message_group_id = f"{self.ctx.trace_id or 'llm'}-{run_uuid}"
        if suffix:
            message_group_id += f"-{suffix}"

        data = {
            "message_group_id": message_group_id,
            "event_type": "chunk",
            "data_format": data_format,
            "index": index,
            "chunk_delta": chunk_delta,
            "is_message_end": is_message_end,
        }
        if isinstance(self._run, dict) and self._run.get("thread_uuid"):
            data["thread_uuid"] = self._run.get("thread_uuid")

        try:
            self.ctx.emit(data)
        except Exception as e:  # noqa: BLE001 — emit must never break the call
            if self.logger and self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning("send_data_to_stream emit failed: %s", e)

    def is_stream_cancelled(self) -> bool:
        """Whether the host cancelled this run (Redis cancel key via
        ``ctx.is_cancelled``). Latches once True so subsequent chunks and the
        generation loop stop. Always False without a context."""
        if self._stream_closed:
            return True
        if self.ctx is not None and callable(getattr(self.ctx, "is_cancelled", None)):
            try:
                if self.ctx.is_cancelled():
                    self._stream_closed = True
                    return True
            except Exception:  # noqa: BLE001 — cancellation probe is best-effort
                return False
        return False

    # ------------------------------------------------------------------
    # Utility functions: JSON completion and chunk buffering (kept as-is)
    # ------------------------------------------------------------------
    def try_complete_json(self, accumulated: str) -> str:
        """Try to validate and complete the JSON by adding closing brackets.

        Args:
            accumulated: The accumulated JSON string that may be incomplete.

        Returns:
            The ending string that successfully completes the JSON, or an
            empty string if no valid completion is found.
        """
        try:
            json.loads(accumulated)  # Attempt to parse with the accumulated
            return accumulated
        except json.JSONDecodeError:
            possible_endings = ["}", "]", "}", "}]", "}]}", "}}"]
            for ending in possible_endings:
                try:
                    completed_json = accumulated + ending
                    json.loads(
                        completed_json
                    )  # Attempt to parse with the potential ending
                    return ending  # If parsing succeeds, return the successful ending
                except json.JSONDecodeError:
                    continue
        return ""  # If no ending works, return empty string

    def process_and_send_json(
        self,
        index: int,
        complete_accumulated_json: str,
        accumulated_partial_json: str,
        data_format: str,
    ) -> tuple[int, str, str]:
        """Process and send JSON if it forms a valid structure.

        Returns:
            tuple: (updated index, complete JSON, remaining partial buffer)
        """
        combined_json = complete_accumulated_json + accumulated_partial_json
        ending = self.try_complete_json(combined_json)

        if ending:
            # Send the accumulated partial JSON to the stream
            self.send_data_to_stream(
                index=index,
                data_format=data_format,
                chunk_delta=accumulated_partial_json,
                is_message_end=False,
            )
            complete_accumulated_json += (
                accumulated_partial_json  # Update complete JSON
            )
            accumulated_partial_json = ""  # Reset the partial JSON accumulator
            index += 1

        return index, complete_accumulated_json, accumulated_partial_json

    def process_text_content(
        self,
        index: int,
        accumulated_partial_text: str,
        output_format: str,
        suffix: str = "",
    ) -> tuple[int, str]:
        """Process XML content from accumulated text and send to stream.

        Returns:
            Tuple of (updated index, remaining text buffer)
        """

        # Check if text contains XML-style tags and update format
        if "<" in accumulated_partial_text:
            output_format = "xml"

            # Wait for closing tag before sending, with max buffer size check
            if ">" in accumulated_partial_text or len(accumulated_partial_text) > 500:
                # If no closing tag found within buffer limit, treat as regular text
                if (
                    ">" not in accumulated_partial_text
                    and len(accumulated_partial_text) > 500
                ):
                    output_format = "text"

                if output_format == "xml":
                    # Extract XML tags and text content
                    xml_parts = []
                    current_text = accumulated_partial_text
                    while "<" in current_text:
                        start_idx = current_text.find("<")
                        if start_idx > 0:
                            xml_parts.append(current_text[:start_idx])
                        end_idx = current_text.find(">", start_idx)
                        if end_idx == -1:
                            xml_parts.append(current_text[start_idx:])
                            current_text = ""
                            break
                        xml_parts.append(current_text[start_idx : end_idx + 1])
                        current_text = current_text[end_idx + 1 :]
                    if current_text:
                        xml_parts.append(current_text)

                    for part in xml_parts:
                        self.send_data_to_stream(
                            index=index,
                            data_format=(
                                output_format if "<" in part and ">" in part else "text"
                            ),
                            chunk_delta=part,
                            suffix=suffix,
                        )
                        index += 1
                else:
                    self.send_data_to_stream(
                        index=index,
                        data_format=output_format,
                        chunk_delta=accumulated_partial_text,
                        suffix=suffix,
                    )
                    index += 1
                accumulated_partial_text = (
                    ""  # For non-XML content, use buffer threshold
                )
        elif len(accumulated_partial_text) >= int(
            self.setting.get("accumulated_partial_text_buffer", "10")
        ):
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
                chunk_delta=accumulated_partial_text,
                suffix=suffix,
            )
            accumulated_partial_text = ""
            index += 1
        return index, accumulated_partial_text
