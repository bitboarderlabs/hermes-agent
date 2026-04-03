"""BotParlor platform adapter.

Connects to BotParlor's WebSocket server and integrates through the
hermes-agent gateway pipeline for full session management, conversation
history, context compression, and memory.

Messages are tagged with turnId and isFinal so BotParlor can:
- Group tool calls with their final response
- Only forward final responses to other bots in group chat
- Only send final responses to TTS
- Display tool calls as indented sub-items

Configuration in config.yaml:
  botparlor:
    url: ws://botparlor.st-el.com:3000/ws
    bot_name: Sarah
    api_key: boobsaregreat
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    build_session_key,
)

logger = logging.getLogger(__name__)

DEFAULT_RECONNECT_DELAY = 5
MAX_RECONNECT_DELAY = 60


def check_botparlor_requirements() -> bool:
    """Check if BotParlor adapter dependencies are available."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            return False
    return True


class BotParlorAdapter(BasePlatformAdapter):
    """
    BotParlor <-> Hermes gateway adapter.

    Connects outbound to BotParlor's WebSocket server. Messages are tagged
    with turnId (groups tool calls with final response) and isFinal (marks
    the actual chat response vs intermediate tool progress).
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.BOTPARLOR)
        extra = config.extra or {}
        self._url: str = extra.get("url", os.getenv("BOTPARLOR_URL", ""))
        self._bot_name: str = extra.get("bot_name", os.getenv("BOTPARLOR_BOT_NAME", ""))
        self._api_key: str = extra.get("api_key", os.getenv("BOTPARLOR_API_KEY", ""))
        self._ws = None
        self._listen_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Turn tracking: groups tool calls with their final response
        self._current_turn_id: Optional[str] = None
        self._current_chat_id: Optional[str] = None
        self._handler_complete = False

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not self._url:
            logger.error("[BotParlor] No URL configured")
            return False
        if not self._bot_name:
            logger.error("[BotParlor] No bot_name configured")
            return False

        try:
            await self._connect_ws()
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._mark_connected()
            logger.info("[BotParlor] Connected as %s to %s", self._bot_name, self._url)
            return True
        except Exception as e:
            logger.error("[BotParlor] Initial connection failed: %s", e)
            self._listen_task = asyncio.create_task(self._listen_loop())
            return True

    async def disconnect(self) -> None:
        self._shutdown = True
        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._mark_disconnected()
        logger.info("[BotParlor] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._ws:
            return SendResult(success=False, error="Not connected to BotParlor")

        # Determine if this is the final response or an intermediate tool call
        is_final = self._handler_complete
        turn_id = self._current_turn_id

        # If no turn is active (e.g. system messages outside handler), generate a one-off
        if not turn_id:
            turn_id = str(uuid.uuid4())[:8]

        try:
            msg = json.dumps({
                "type": "bot.response",
                "payload": {
                    "botName": self._bot_name,
                    "chatId": self._current_chat_id or chat_id,
                    "text": content,
                    "turnId": turn_id,
                    "isFinal": is_final,
                },
            })
            await self._ws.send(msg)
            return SendResult(success=True)
        except Exception as e:
            logger.error("[BotParlor] send failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if not self._ws:
            return
        try:
            msg = json.dumps({
                "type": "bot.typing",
                "payload": {
                    "botName": self._bot_name,
                    "chatId": self._current_chat_id or chat_id,
                },
            })
            await self._ws.send(msg)
        except Exception:
            pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        if chat_id.startswith("group:"):
            return {"name": chat_id, "type": "group", "chat_id": chat_id}
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    # ------------------------------------------------------------------
    # Override _process_message_background for turn tracking
    # ------------------------------------------------------------------

    async def _process_message_background(self, event: MessageEvent, session_key: str) -> None:
        """Override to track turn lifecycle for isFinal tagging."""
        # Start a new turn
        self._current_turn_id = str(uuid.uuid4())[:8]
        self._current_chat_id = event.source.chat_id
        self._handler_complete = False

        # Create interrupt event for this session
        interrupt_event = asyncio.Event()
        self._active_sessions[session_key] = interrupt_event

        # Start typing indicator
        _thread_metadata = {"thread_id": event.source.thread_id} if event.source.thread_id else None
        typing_task = asyncio.create_task(self._keep_typing(event.source.chat_id, metadata=_thread_metadata))

        try:
            # Call the handler (tool calls send() during this)
            response = await self._message_handler(event)

            # Mark handler as complete — next send() will be tagged isFinal
            self._handler_complete = True

            # Send the final response
            if response:
                # Extract images/media (base class logic)
                media_files, response = self.extract_media(response)
                images, text_content = self.extract_images(response)
                text_content = text_content.replace("[[audio_as_voice]]", "").strip()
                import re
                text_content = re.sub(r"MEDIA:\s*\S+", "", text_content).strip()
                local_files, text_content = self.extract_local_files(text_content)

                if text_content:
                    await self.send(event.source.chat_id, text_content)

                # Send media files
                for path in media_files:
                    await self._send_media_file(event.source.chat_id, path)
                for url in images:
                    try:
                        await self.send_image(event.source.chat_id, url)
                    except Exception:
                        pass
                for path in local_files:
                    await self._send_local_file(event.source.chat_id, path)

        except Exception as e:
            logger.error("[BotParlor] Handler error: %s", e, exc_info=True)
            self._handler_complete = True
            await self.send(event.source.chat_id, f"Error: {e}")
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

            # Clean up session tracking
            self._active_sessions.pop(session_key, None)
            self._current_turn_id = None
            self._current_chat_id = None
            self._handler_complete = False

            # Process pending message if any (interrupt support)
            pending = self._pending_messages.pop(session_key, None)
            if pending:
                if isinstance(pending, MessageEvent):
                    asyncio.create_task(self._process_message_background(pending, session_key))
                elif isinstance(pending, str):
                    # Text from interrupt — create a new event
                    source = event.source
                    new_event = MessageEvent(
                        text=pending,
                        message_type=MessageType.COMMAND if pending.startswith("/") else MessageType.TEXT,
                        source=source,
                        raw_message=None,
                    )
                    asyncio.create_task(self._process_message_background(new_event, session_key))

    async def _send_media_file(self, chat_id: str, path: str) -> None:
        """Send a media file to BotParlor."""
        pass  # TODO: implement if needed

    async def _send_local_file(self, chat_id: str, path: str) -> None:
        """Send a local file to BotParlor."""
        pass  # TODO: implement if needed

    # ------------------------------------------------------------------
    # WebSocket connection and message handling
    # ------------------------------------------------------------------

    async def _connect_ws(self) -> None:
        """Open WebSocket connection to BotParlor."""
        try:
            import websockets
            self._ws = await websockets.connect(self._url)
        except ImportError:
            import aiohttp
            session = aiohttp.ClientSession()
            self._ws = await session.ws_connect(self._url)

        hello = json.dumps({
            "type": "bot.hello",
            "payload": {
                "botName": self._bot_name,
                "apiKey": self._api_key,
            },
        })
        await self._ws.send(hello)
        logger.info("[BotParlor] WebSocket connected, sent hello as %s", self._bot_name)

    async def _listen_loop(self) -> None:
        """Main loop: listen for messages, reconnect on disconnect."""
        delay = DEFAULT_RECONNECT_DELAY

        while not self._shutdown:
            try:
                if not self._ws:
                    await self._connect_ws()
                    self._mark_connected()
                    delay = DEFAULT_RECONNECT_DELAY
                    logger.info("[BotParlor] Reconnected as %s", self._bot_name)

                async for raw in self._ws:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    try:
                        data = json.loads(raw)
                        await self._handle_bp_message(data)
                    except json.JSONDecodeError:
                        logger.warning("[BotParlor] Invalid JSON: %s", raw[:100])
                    except Exception as e:
                        logger.error("[BotParlor] Message handling error: %s", e, exc_info=True)

                logger.info("[BotParlor] Connection closed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[BotParlor] Connection error: %s — reconnecting in %ds", e, delay)

            self._ws = None
            self._mark_disconnected()

            if self._shutdown:
                break

            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_RECONNECT_DELAY)

    async def _handle_bp_message(self, data: dict) -> None:
        """Process a message from BotParlor."""
        msg_type = data.get("type", "")
        payload = data.get("payload", {})

        if msg_type == "chat.incoming":
            chat_id = payload.get("chatId", "")
            text = payload.get("text", "")
            sender = payload.get("sender", "User")

            if not text:
                return

            chat_type = "group" if chat_id.startswith("group:") else "dm"

            source = self.build_source(
                chat_id=chat_id,
                chat_name=chat_id,
                chat_type=chat_type,
                user_id=sender,
                user_name=sender,
            )

            evt_type = MessageType.COMMAND if text.startswith("/") else MessageType.TEXT

            event = MessageEvent(
                text=text,
                message_type=evt_type,
                source=source,
                raw_message=data,
            )

            await self.handle_message(event)

        elif msg_type == "ping":
            if self._ws:
                await self._ws.send(json.dumps({"type": "pong"}))
