"""
OpenAI-compatible API server platform adapter.

Exposes an HTTP server with endpoints:
- POST /v1/chat/completions        — OpenAI Chat Completions format (stateless)
- POST /v1/responses               — OpenAI Responses API format (stateful via previous_response_id)
- GET  /v1/responses/{response_id} — Retrieve a stored response
- DELETE /v1/responses/{response_id} — Delete a stored response
- GET  /v1/models                  — lists hermes-agent as an available model
- GET  /health                     — health check

Any OpenAI-compatible frontend (Open WebUI, LobeChat, LibreChat,
AnythingLLM, NextChat, ChatBox, etc.) can connect to hermes-agent
through this adapter by pointing at http://localhost:8642/v1.

Requires:
- aiohttp (already available in the gateway)
"""

import asyncio
import collections
import concurrent.futures
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
MAX_STORED_RESPONSES = 100


def check_api_server_requirements() -> bool:
    """Check if API server dependencies are available."""
    return AIOHTTP_AVAILABLE


class ResponseStore:
    """
    In-memory LRU store for Responses API state.

    Each stored response includes the full internal conversation history
    (with tool calls and results) so it can be reconstructed on subsequent
    requests via previous_response_id.
    """

    def __init__(self, max_size: int = MAX_STORED_RESPONSES):
        self._store: collections.OrderedDict[str, Dict[str, Any]] = collections.OrderedDict()
        self._max_size = max_size

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID (moves to end for LRU)."""
        if response_id in self._store:
            self._store.move_to_end(response_id)
            return self._store[response_id]
        return None

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        """Store a response, evicting the oldest if at capacity."""
        if response_id in self._store:
            self._store.move_to_end(response_id)
        self._store[response_id] = data
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def delete(self, response_id: str) -> bool:
        """Remove a response from the store. Returns True if found and deleted."""
        if response_id in self._store:
            del self._store[response_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def cors_middleware(request, handler):
        """Add CORS headers to every response; handle OPTIONS preflight."""
        if request.method == "OPTIONS":
            return web.Response(status=200, headers=_CORS_HEADERS)
        response = await handler(request)
        response.headers.update(_CORS_HEADERS)
        return response
else:
    cors_middleware = None  # type: ignore[assignment]


# ===================================================================
# BotParlor WebSocket session (OpenClaw protocol v4)
# ===================================================================

TICK_INTERVAL_S = 25
SERVER_VERSION = "1.0.0"

# Regex to parse tool progress messages: "💻 terminal: \"ls -la\""
_TOOL_PROGRESS_RE = re.compile(
    r'^.{1,4}\s+(\w+)(?::\s*"(.+)"|\.\.\.)?\s*(?:\(×(\d+)\))?$',
    re.DOTALL,
)

# Regex to extract <think>...</think> blocks
_THINK_RE = re.compile(r'<think>([\s\S]*?)</think>', re.IGNORECASE)

# Regex to extract set_mood("mood") commands
_MOOD_RE = re.compile(r'set_?mood\s*\(\s*["\']([^"\']+)["\']\s*\)', re.IGNORECASE)


def _gen_id() -> str:
    return uuid.uuid4().hex[:16]


class _BotParlorSession:
    """Manages one BotParlor WebSocket connection."""

    def __init__(self, ws: "web.WebSocketResponse", adapter: "APIServerAdapter"):
        self.ws = ws
        self.adapter = adapter
        self.conn_id = _gen_id()
        self.authenticated = False
        self._tick_task: Optional[asyncio.Task] = None
        self._active_runs: Dict[str, asyncio.Task] = {}
        self._tool_seq: int = 0

    # -- frame helpers --------------------------------------------------

    async def _send_event(self, event: str, payload: Any = None) -> None:
        frame: dict = {"type": "event", "event": event}
        if payload is not None:
            frame["payload"] = payload
        try:
            await self.ws.send_str(json.dumps(frame))
        except Exception:
            pass

    async def _send_response(
        self, req_id: str, ok: bool, payload: Any = None, error: dict | None = None
    ) -> None:
        frame: dict = {"type": "res", "id": req_id, "ok": ok}
        if payload is not None:
            frame["payload"] = payload
        if error is not None:
            frame["error"] = error
        try:
            await self.ws.send_str(json.dumps(frame))
        except Exception:
            pass

    # -- main loop ------------------------------------------------------

    async def run(self) -> None:
        await self._send_event("connect.challenge", {
            "nonce": _gen_id(),
            "ts": int(time.time() * 1000),
        })

        try:
            async for msg in self.ws:
                if msg.type in (web.WSMsgType.TEXT,):
                    await self._handle_frame(msg.data)
                elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                    break
        except Exception as e:
            logger.error("[WS] Session error (conn=%s): %s", self.conn_id, e)
        finally:
            await self._cleanup()

    async def _handle_frame(self, data: str) -> None:
        try:
            frame = json.loads(data)
        except json.JSONDecodeError:
            return

        if frame.get("type") != "req":
            return

        req_id = frame.get("id", "")
        method = frame.get("method", "")
        params = frame.get("params", {})

        try:
            if method == "connect":
                result = await self._handle_connect(params)
            elif not self.authenticated:
                await self._send_response(req_id, False, error={
                    "code": "NOT_AUTHENTICATED",
                    "message": "Must connect first",
                })
                return
            elif method == "chat.send":
                result = await self._handle_chat_send(req_id, params)
            elif method == "chat.history":
                result = self._handle_chat_history(params)
            elif method == "chat.abort":
                result = self._handle_chat_abort(params)
            elif method == "sensor.result":
                result = await self._handle_sensor_result(params)
            elif method == "sessions.reset":
                result = {"ok": True}
            elif method == "sessions.delete":
                result = {"ok": True}
            elif method == "health":
                result = {"status": "ok"}
            elif method == "agent.identity.get":
                result = self._handle_identity_get()
            else:
                await self._send_response(req_id, False, error={
                    "code": "UNKNOWN_METHOD",
                    "message": f"Unknown method: {method}",
                })
                return

            await self._send_response(req_id, True, payload=result)

        except Exception as e:
            logger.error("[WS] Error handling %s: %s", method, e, exc_info=True)
            await self._send_response(req_id, False, error={
                "code": "INTERNAL_ERROR",
                "message": str(e),
            })

    # -- method handlers ------------------------------------------------

    async def _handle_connect(self, params: dict) -> dict:
        auth = params.get("auth", {})
        token = auth.get("token", "")
        if self.adapter._api_key and token != self.adapter._api_key:
            raise Exception("Invalid auth token")

        self.authenticated = True
        self._tick_task = asyncio.create_task(self._tick_loop())

        client_name = params.get("client", {}).get("displayName", "unknown")
        logger.info("[WS] Client authenticated: %s (conn=%s)", client_name, self.conn_id)

        return {
            "protocol": 4,
            "server": {
                "version": SERVER_VERSION,
                "connId": self.conn_id,
                "backend": "hermes-agent",
                "features": {
                    "thinking": True,
                    "toolProgress": True,
                    "subAgents": True,
                    "executeCode": True,
                    "streaming": True,
                    "memory": True,
                    "skills": True,
                },
            },
            "methods": [
                "connect", "chat.send", "chat.history", "chat.abort",
                "sessions.reset", "sessions.delete", "health",
                "agent.identity.get",
            ],
            "events": [
                "chat", "agent.thinking", "agent.tool.start",
                "agent.tool.result", "agent.progress",
                "display.mood", "display.media", "display.toast",
                "tick", "shutdown",
            ],
            "policy": {
                "tickIntervalMs": TICK_INTERVAL_S * 1000,
            },
        }

    async def _handle_chat_send(self, req_id: str, params: dict) -> dict:
        session_key = params.get("sessionKey", "agent:main:main")
        message = params.get("message", "")
        run_id = _gen_id()

        # Do NOT strip slash commands here.  Let them pass through to the
        # gateway's handle_message so command dispatch (_handle_reset_command
        # for /new, etc.) fires correctly.  Previously /new was silently
        # stripped, causing the session to never actually reset.
        task = asyncio.create_task(
            self._run_agent(session_key, run_id, message)
        )
        self._active_runs[run_id] = task

        return {"runId": run_id, "status": "started"}

    async def _run_agent(self, session_key: str, run_id: str, message: str) -> None:
        """Run the agent and stream structured events back via WebSocket."""
        self._tool_seq = 0
        chat_id = f"ws-{session_key}"

        try:
            await asyncio.sleep(0.15)

            if not self.adapter._message_handler:
                raise Exception("No message handler configured")

            source = self.adapter.build_source(
                chat_id=chat_id,
                chat_name=f"BotParlor ({session_key})",
                chat_type="dm",
                user_id="botparlor-user",
                user_name="BotParlor",
            )

            event = MessageEvent(
                text=message,
                message_type=MessageType.TEXT,
                source=source,
                message_id=run_id,
            )

            # Register this session for progress routing
            self.adapter._register_ws_session(chat_id, self, run_id)

            try:
                response_text = await self.adapter._message_handler(event)
            finally:
                self.adapter._unregister_ws_session(chat_id)

            if not response_text:
                response_text = ""

            # Consume the structured agent result stored by GatewayRunner
            agent_result = self.adapter._consume_agent_result(chat_id)

            # --- Build final message content blocks ---

            content_blocks = []

            # 1. Thinking/reasoning from agent result
            if agent_result and agent_result.get("last_reasoning"):
                content_blocks.append({
                    "type": "thinking",
                    "thinking": agent_result["last_reasoning"].strip(),
                })

            # Fallback: extract <think> tags from response text
            if not any(b["type"] == "thinking" for b in content_blocks):
                think_matches = _THINK_RE.findall(response_text)
                if think_matches:
                    thinking_text = "\n\n".join(t.strip() for t in think_matches if t.strip())
                    if thinking_text:
                        content_blocks.append({"type": "thinking", "thinking": thinking_text})

            # Strip thinking tags from display text
            response_text = _THINK_RE.sub("", response_text).strip()

            # 2. Extract tool calls from agent transcript
            if agent_result and agent_result.get("messages"):
                for msg in agent_result["messages"]:
                    role = msg.get("role", "")
                    if role == "assistant" and msg.get("tool_calls"):
                        for tc in msg["tool_calls"]:
                            fn = tc.get("function", {})
                            tc_name = fn.get("name", "")
                            tc_id = tc.get("id", "")
                            try:
                                tc_args = json.loads(fn.get("arguments", "{}"))
                            except (json.JSONDecodeError, TypeError):
                                tc_args = fn.get("arguments", "")
                            content_blocks.append({
                                "type": "toolCall",
                                "id": tc_id,
                                "name": tc_name,
                                "input": tc_args,
                            })
                    elif role == "tool":
                        tc_id = msg.get("tool_call_id", "")
                        tc_content = msg.get("content", "")
                        if len(tc_content) > 2000:
                            tc_content = tc_content[:1997] + "..."
                        content_blocks.append({
                            "type": "toolResult",
                            "toolUseId": tc_id,
                            "content": tc_content,
                        })

            # 3. Add text content
            if response_text:
                content_blocks.append({"type": "text", "text": response_text})

            # 4. Send final response with all content blocks
            await self._send_event("chat", {
                "runId": run_id,
                "sessionKey": session_key,
                "seq": self._tool_seq + 1,
                "state": "final",
                "message": {
                    "role": "assistant",
                    "content": content_blocks,
                },
                "usage": {
                    "inputTokens": agent_result.get("last_prompt_tokens", 0) if agent_result else 0,
                    "outputTokens": 0,
                },
            })

        except asyncio.CancelledError:
            await self._send_event("chat", {
                "runId": run_id,
                "sessionKey": session_key,
                "seq": self._tool_seq + 1,
                "state": "aborted",
            })
        except Exception as e:
            logger.error("[WS] Agent run error: %s", e, exc_info=True)
            await self._send_event("chat", {
                "runId": run_id,
                "sessionKey": session_key,
                "seq": self._tool_seq + 1,
                "state": "error",
                "errorMessage": str(e),
            })
        finally:
            self._active_runs.pop(run_id, None)

    async def emit_tool_progress(self, run_id: str, text: str) -> None:
        """Parse a tool progress message and emit structured event."""
        self._tool_seq += 1
        first_line = text.split("\n")[0] if "\n" in text else text
        match = _TOOL_PROGRESS_RE.match(first_line)

        if match:
            tool_name = match.group(1)
            preview = match.group(2) or ""
            repeat = match.group(3)

            if repeat:
                await self._send_event("agent.progress", {
                    "runId": run_id,
                    "tool": tool_name,
                    "repeat": int(repeat),
                    "text": first_line,
                })
            else:
                await self._send_event("agent.tool.start", {
                    "runId": run_id,
                    "toolCallId": f"tc_{self._tool_seq}",
                    "name": tool_name,
                    "preview": preview,
                    "iteration": self._tool_seq,
                })
        else:
            await self._send_event("agent.progress", {
                "runId": run_id,
                "text": text,
            })

    def _handle_chat_history(self, params: dict) -> dict:
        return {"messages": []}

    def _handle_chat_abort(self, params: dict) -> dict:
        run_id = params.get("runId", "")
        task = self._active_runs.get(run_id)
        if task and not task.done():
            task.cancel()
            logger.info("[WS] Aborted run %s", run_id)
        return {"ok": True}

    def _handle_identity_get(self) -> dict:
        return {"name": "Hermes Agent", "role": "AI Assistant"}

    # -- BotParlor tool callback (called from agent thread) -------------

    _pending_sensors: Dict[str, concurrent.futures.Future] = {}

    def handle_botparlor_tool(self, action: str, params: dict) -> dict:
        """Handle a BotParlor tool call from the agent.

        Called from the agent's thread (synchronous). For display actions
        (set_mood, display_media), we fire-and-forget an async event.
        For sensor requests (camera, location), we block until the client
        responds via the WebSocket.
        """
        loop = self.adapter._get_event_loop()
        if not loop:
            return {"error": "No event loop available"}

        if action == "set_mood":
            asyncio.run_coroutine_threadsafe(
                self._send_event("display.mood", {
                    "mood": params.get("mood", "default"),
                }),
                loop,
            ).result(timeout=5)
            return {"ok": True, "mood": params["mood"]}

        elif action == "display_media":
            asyncio.run_coroutine_threadsafe(
                self._send_event("display.media", {
                    "media": {
                        "id": _gen_id(),
                        "type": params.get("type", "image"),
                        "url": params.get("url", ""),
                        "title": params.get("title", ""),
                    },
                    "display": {
                        "mode": params.get("mode", "panel"),
                        "panelId": params.get("panelId"),
                        "force": params.get("force", False),
                    },
                }),
                loop,
            ).result(timeout=5)
            return {"ok": True}

        elif action == "sensor_request":
            return self._handle_sensor_request(params, loop)

        return {"error": f"Unknown BotParlor action: {action}"}

    def _handle_sensor_request(self, params: dict, loop: asyncio.AbstractEventLoop) -> dict:
        """Block until client responds to sensor request."""
        request_id = _gen_id()
        sensor_type = params.get("sensorType", "camera")
        timeout = 60 if sensor_type == "camera" else 15

        future: concurrent.futures.Future = concurrent.futures.Future()
        self._pending_sensors[request_id] = future

        try:
            asyncio.run_coroutine_threadsafe(
                self._send_event("sensor.request", {
                    "requestId": request_id,
                    "sensorType": sensor_type,
                    "params": {k: v for k, v in params.items() if k != "sensorType"},
                    "timeout": timeout * 1000,
                }),
                loop,
            ).result(timeout=5)

            result = future.result(timeout=timeout)
            return result

        except concurrent.futures.TimeoutError:
            logger.warning("[WS] Sensor request %s timed out after %ds", request_id, timeout)
            return {"error": f"Sensor request timed out ({sensor_type})."}
        except Exception as e:
            logger.error("[WS] Sensor request error: %s", e)
            return {"error": str(e)}
        finally:
            self._pending_sensors.pop(request_id, None)

    async def _handle_sensor_result(self, params: dict) -> dict:
        """Handle sensor.result from the WebSocket client."""
        request_id = params.get("requestId", "")
        future = self._pending_sensors.get(request_id)
        if not future:
            return {"ok": False, "error": "Unknown request ID"}

        if params.get("success"):
            future.set_result(params.get("data", {}))
        else:
            error = params.get("error", {})
            future.set_result({
                "error": error.get("message", "Sensor request failed"),
                "code": error.get("code", "unknown"),
            })

        return {"ok": True}

    async def _tick_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(TICK_INTERVAL_S)
                await self._send_event("tick")
        except (asyncio.CancelledError, Exception):
            pass

    async def _cleanup(self) -> None:
        if self._tick_task:
            self._tick_task.cancel()
        for run_id, task in list(self._active_runs.items()):
            task.cancel()
        logger.info("[WS] Client disconnected (conn=%s)", self.conn_id)


# ===================================================================
# Main adapter
# ===================================================================

class APIServerAdapter(BasePlatformAdapter):
    """
    OpenAI-compatible HTTP API server adapter.

    Runs an aiohttp web server that accepts OpenAI-format requests
    and routes them through hermes-agent's AIAgent.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.API_SERVER)
        extra = config.extra or {}
        self._host: str = extra.get("host", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
        self._port: int = int(extra.get("port", os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))))
        self._api_key: str = extra.get("key", os.getenv("API_SERVER_KEY", ""))
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._response_store = ResponseStore()
        # Conversation name → latest response_id mapping
        self._conversations: Dict[str, str] = {}
        # BotParlor WebSocket state
        self._ws_sessions: Dict[str, tuple] = {}  # {chat_id: (session, run_id)}
        self._agent_results: Dict[str, dict] = {}  # {chat_id: result_dict}

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """
        Validate Bearer token from Authorization header.

        Returns None if auth is OK, or a 401 web.Response on failure.
        If no API key is configured, all requests are allowed.
        """
        if not self._api_key:
            return None  # No key configured — allow all (local-only use)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if token == self._api_key:
                return None  # Auth OK

        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status=401,
        )

    # ------------------------------------------------------------------
    # Agent creation helper
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
    ) -> Any:
        """
        Create an AIAgent instance using the gateway's runtime config.

        Uses _resolve_runtime_agent_kwargs() to pick up model, api_key,
        base_url, etc. from config.yaml / env vars.
        """
        from run_agent import AIAgent
        from gateway.run import _resolve_runtime_agent_kwargs, _resolve_gateway_model

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        model = _resolve_gateway_model()

        max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))

        agent = AIAgent(
            model=model,
            **runtime_kwargs,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt=ephemeral_system_prompt or None,
            session_id=session_id,
            platform="api_server",
            stream_delta_callback=stream_delta_callback,
        )
        return agent

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "hermes-agent"})

    async def _handle_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models — return hermes-agent as an available model."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": "hermes-agent",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "hermes",
                    "permission": [],
                    "root": "hermes-agent",
                    "parent": None,
                }
            ],
        })

    async def _handle_chat_completions(self, request: "web.Request") -> "web.Response":
        """POST /v1/chat/completions — OpenAI Chat Completions format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return web.json_response(
                {"error": {"message": "Missing or invalid 'messages' field", "type": "invalid_request_error"}},
                status=400,
            )

        stream = body.get("stream", False)

        # Extract system message (becomes ephemeral system prompt layered ON TOP of core)
        system_prompt = None
        conversation_messages: List[Dict[str, str]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                # Accumulate system messages
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt = system_prompt + "\n" + content
            elif role in ("user", "assistant"):
                conversation_messages.append({"role": role, "content": content})

        # Extract the last user message as the primary input
        user_message = ""
        history = []
        if conversation_messages:
            user_message = conversation_messages[-1].get("content", "")
            history = conversation_messages[:-1]

        if not user_message:
            return web.json_response(
                {"error": {"message": "No user message found in messages", "type": "invalid_request_error"}},
                status=400,
            )

        session_id = str(uuid.uuid4())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        model_name = body.get("model", "hermes-agent")
        created = int(time.time())

        if stream:
            import queue as _q
            _stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                _stream_q.put(delta)

            # Start agent in background
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                stream_delta_callback=_on_delta,
            ))

            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, _stream_q, agent_task
            )

        # Non-streaming: run the agent and return full response
        try:
            result, usage = await self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Error running agent for chat completions: %s", e, exc_info=True)
            return web.json_response(
                {"error": {"message": f"Internal server error: {e}", "type": "server_error"}},
                status=500,
            )

        final_response = result.get("final_response", "")
        if not final_response:
            final_response = result.get("error", "(No response generated)")

        response_data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_response,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        return web.json_response(response_data)

    async def _write_sse_chat_completion(
        self, request: "web.Request", completion_id: str, model: str,
        created: int, stream_q, agent_task,
    ) -> "web.StreamResponse":
        """Write real streaming SSE from agent's stream_delta_callback queue."""
        import queue as _q

        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await response.prepare(request)

        # Role chunk
        role_chunk = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        await response.write(f"data: {json.dumps(role_chunk)}\n\n".encode())

        # Stream content chunks as they arrive from the agent
        loop = asyncio.get_event_loop()
        while True:
            try:
                delta = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
            except _q.Empty:
                if agent_task.done():
                    # Drain any remaining items
                    while True:
                        try:
                            delta = stream_q.get_nowait()
                            if delta is None:
                                break
                            content_chunk = {
                                "id": completion_id, "object": "chat.completion.chunk",
                                "created": created, "model": model,
                                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                            }
                            await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())
                        except _q.Empty:
                            break
                    break
                continue

            if delta is None:  # End of stream sentinel
                break

            content_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())

        # Get usage from completed agent
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        try:
            result, agent_usage = await agent_task
            usage = agent_usage or usage
        except Exception:
            pass

        # Finish chunk
        finish_chunk = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")

        return response

    async def _handle_responses(self, request: "web.Request") -> "web.Response":
        """POST /v1/responses — OpenAI Responses API format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        raw_input = body.get("input")
        if raw_input is None:
            return web.json_response(
                {"error": {"message": "Missing 'input' field", "type": "invalid_request_error"}},
                status=400,
            )

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        conversation = body.get("conversation")
        store = body.get("store", True)

        # conversation and previous_response_id are mutually exclusive
        if conversation and previous_response_id:
            return web.json_response(
                {"error": {"message": "Cannot use both 'conversation' and 'previous_response_id'", "type": "invalid_request_error"}},
                status=400,
            )

        # Resolve conversation name to latest response_id
        if conversation:
            previous_response_id = self._conversations.get(conversation)
            # No error if conversation doesn't exist yet — it's a new conversation

        # Normalize input to message list
        input_messages: List[Dict[str, str]] = []
        if isinstance(raw_input, str):
            input_messages = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            for item in raw_input:
                if isinstance(item, str):
                    input_messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    # Handle content that may be a list of content parts
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "input_text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, dict) and part.get("type") == "output_text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        content = "\n".join(text_parts)
                    input_messages.append({"role": role, "content": content})
        else:
            return web.json_response(
                {"error": {"message": "'input' must be a string or array", "type": "invalid_request_error"}},
                status=400,
            )

        # Reconstruct conversation history from previous_response_id
        conversation_history: List[Dict[str, str]] = []
        if previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored is None:
                return web.json_response(
                    {"error": {"message": f"Previous response not found: {previous_response_id}", "type": "invalid_request_error"}},
                    status=404,
                )
            conversation_history = list(stored.get("conversation_history", []))
            # If no instructions provided, carry forward from previous
            if instructions is None:
                instructions = stored.get("instructions")

        # Append new input messages to history (all but the last become history)
        for msg in input_messages[:-1]:
            conversation_history.append(msg)

        # Last input message is the user_message
        user_message = input_messages[-1].get("content", "") if input_messages else ""
        if not user_message:
            return web.json_response(
                {"error": {"message": "No user message found in input", "type": "invalid_request_error"}},
                status=400,
            )

        # Truncation support
        if body.get("truncation") == "auto" and len(conversation_history) > 100:
            conversation_history = conversation_history[-100:]

        # Run the agent
        session_id = str(uuid.uuid4())
        try:
            result, usage = await self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Error running agent for responses: %s", e, exc_info=True)
            return web.json_response(
                {"error": {"message": f"Internal server error: {e}", "type": "server_error"}},
                status=500,
            )

        final_response = result.get("final_response", "")
        if not final_response:
            final_response = result.get("error", "(No response generated)")

        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        created_at = int(time.time())

        # Build the full conversation history for storage
        # (includes tool calls from the agent run)
        full_history = list(conversation_history)
        full_history.append({"role": "user", "content": user_message})
        # Add agent's internal messages if available
        agent_messages = result.get("messages", [])
        if agent_messages:
            full_history.extend(agent_messages)
        else:
            full_history.append({"role": "assistant", "content": final_response})

        # Build output items (includes tool calls + final message)
        output_items = self._extract_output_items(result)

        response_data = {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "created_at": created_at,
            "model": body.get("model", "hermes-agent"),
            "output": output_items,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        # Store the complete response object for future chaining / GET retrieval
        if store:
            self._response_store.put(response_id, {
                "response": response_data,
                "conversation_history": full_history,
                "instructions": instructions,
            })
            # Update conversation mapping so the next request with the same
            # conversation name automatically chains to this response
            if conversation:
                self._conversations[conversation] = response_id

        return web.json_response(response_data)

    # ------------------------------------------------------------------
    # GET / DELETE response endpoints
    # ------------------------------------------------------------------

    async def _handle_get_response(self, request: "web.Request") -> "web.Response":
        """GET /v1/responses/{response_id} — retrieve a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        stored = self._response_store.get(response_id)
        if stored is None:
            return web.json_response(
                {"error": {"message": f"Response not found: {response_id}", "type": "invalid_request_error"}},
                status=404,
            )

        return web.json_response(stored["response"])

    async def _handle_delete_response(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/responses/{response_id} — delete a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        deleted = self._response_store.delete(response_id)
        if not deleted:
            return web.json_response(
                {"error": {"message": f"Response not found: {response_id}", "type": "invalid_request_error"}},
                status=404,
            )

        return web.json_response({
            "id": response_id,
            "object": "response",
            "deleted": True,
        })

    # ------------------------------------------------------------------
    # Output extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_output_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build the full output item array from the agent's messages.

        Walks *result["messages"]* and emits:
        - ``function_call`` items for each tool_call on assistant messages
        - ``function_call_output`` items for each tool-role message
        - a final ``message`` item with the assistant's text reply
        """
        items: List[Dict[str, Any]] = []
        messages = result.get("messages", [])

        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    items.append({
                        "type": "function_call",
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                        "call_id": tc.get("id", ""),
                    })
            elif role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        # Final assistant message
        final = result.get("final_response", "")
        if not final:
            final = result.get("error", "(No response generated)")

        items.append({
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": final,
                }
            ],
        })
        return items

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
    ) -> tuple:
        """
        Create an agent and run a conversation in a thread executor.

        Returns ``(result_dict, usage_dict)`` where *usage_dict* contains
        ``input_tokens``, ``output_tokens`` and ``total_tokens``.
        """
        loop = asyncio.get_event_loop()

        def _run():
            agent = self._create_agent(
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                stream_delta_callback=stream_delta_callback,
            )
            result = agent.run_conversation(
                user_message=user_message,
                conversation_history=conversation_history,
            )
            usage = {
                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            }
            return result, usage

        return await loop.run_in_executor(None, _run)

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the aiohttp web server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False

        try:
            self._app = web.Application(middlewares=[cors_middleware])
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/v1/models", self._handle_models)
            self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
            self._app.router.add_post("/v1/responses", self._handle_responses)
            self._app.router.add_get("/v1/responses/{response_id}", self._handle_get_response)
            self._app.router.add_delete("/v1/responses/{response_id}", self._handle_delete_response)

            # BotParlor WebSocket endpoint (OpenClaw protocol v4)
            self._app.router.add_get("/ws", self._handle_ws)
            self._app.router.add_get("/ws/{agent_name}", self._handle_ws)

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()

            # Store event loop reference for thread→async bridging (BotParlor)
            self._event_loop = asyncio.get_event_loop()

            self._mark_connected()
            logger.info(
                "[%s] API server listening on http://%s:%d  (REST: /v1, WebSocket: /ws)",
                self.name, self._host, self._port,
            )
            return True

        except Exception as e:
            logger.error("[%s] Failed to start API server: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the aiohttp web server."""
        self._mark_disconnected()
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        logger.info("[%s] API server stopped", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Intercept messages from the GatewayRunner.

        During agent execution, the GatewayRunner sends tool progress
        messages via this method. If a BotParlor WebSocket session is active
        for this chat_id, route them as structured agent.tool.start events.
        For REST API requests, this is a no-op (HTTP handles delivery directly).
        """
        ws_entry = self._ws_sessions.get(chat_id)
        if ws_entry:
            session, run_id = ws_entry
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        session.emit_tool_progress(run_id, content)
                    )
                else:
                    await session.emit_tool_progress(run_id, content)
            except Exception as e:
                logger.debug("[ApiServer] Failed to emit tool progress: %s", e)
            return SendResult(success=True, message_id=_gen_id())

        return SendResult(success=True)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        """
        Intercept message edits from the GatewayRunner.

        The progress system tries to edit a single message with
        accumulated tool lines. We emit the latest tool as a new event.
        """
        ws_entry = self._ws_sessions.get(chat_id)
        if ws_entry:
            session, run_id = ws_entry
            lines = content.strip().split("\n")
            last_line = lines[-1] if lines else content
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        session.emit_tool_progress(run_id, last_line)
                    )
                else:
                    await session.emit_tool_progress(run_id, last_line)
            except Exception as e:
                logger.debug("[ApiServer] Failed to emit tool progress edit: %s", e)
            return SendResult(success=True, message_id=message_id)

        return SendResult(success=False, error="Edit not supported")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the API server."""
        return {
            "name": "API Server",
            "type": "api",
            "host": self._host,
            "port": self._port,
        }

    # ------------------------------------------------------------------
    # BotParlor WebSocket support
    # ------------------------------------------------------------------

    def _store_agent_result(self, chat_id: str, result: dict) -> None:
        """Store structured agent result for the WebSocket session to consume."""
        self._agent_results[chat_id] = result

    def _consume_agent_result(self, chat_id: str) -> Optional[dict]:
        """Consume and return the stored agent result (one-shot)."""
        return self._agent_results.pop(chat_id, None)

    def _register_ws_session(
        self, chat_id: str, session: _BotParlorSession, run_id: str
    ) -> None:
        self._ws_sessions[chat_id] = (session, run_id)

    def _unregister_ws_session(self, chat_id: str) -> None:
        self._ws_sessions.pop(chat_id, None)

    def _get_event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Return the running event loop for async bridging."""
        return getattr(self, '_event_loop', None)

    def get_botparlor_callback(self, chat_id: str):
        """Return a tool callback for the active WebSocket session, if any."""
        ws_entry = self._ws_sessions.get(chat_id)
        if not ws_entry:
            return None
        session, _run_id = ws_entry
        return session.handle_botparlor_tool

    async def _handle_ws(self, request: "web.Request") -> "web.WebSocketResponse":
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)

        agent_name = request.match_info.get("agent_name")
        logger.info("[WS] New connection (agent=%s)", agent_name or "default")

        session = _BotParlorSession(ws, self)
        await session.run()

        return ws
