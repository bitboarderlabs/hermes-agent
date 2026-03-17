"""
BotParlor integration tools for Hermes Agent.

These tools allow the agent to interact with BotParlor client features:
  - list_moods: Fetch available avatar moods from BotParlor manifest
  - set_mood: Change avatar mood/emotion
  - display_media: Display media in the client's media panel
  - take_photo: Request a photo from the client's camera
  - get_location: Request GPS coordinates from the client

Tools communicate with BotParlor via a callback injected by the gateway
adapter. The callback is set on the AIAgent instance as
`botparlor_callback` and follows the signature:

    callback(action: str, params: dict) -> dict

Actions:
    "list_moods"     -> {"bot": "...", "moods": [...]}
    "set_mood"       -> {"ok": True}
    "display_media"  -> {"ok": True}
    "sensor_request" -> {"url": "...", "width": ..., ...} or {"error": "..."}
"""

import json
import logging
import os
import requests
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _botparlor_call(
    action: str,
    params: dict,
    callback: Optional[Callable] = None,
) -> str:
    """Dispatch a BotParlor action via the injected callback."""
    if callback is None:
        return json.dumps({
            "error": "BotParlor tools are not available in this context. "
                     "These tools only work when connected via the BotParlor gateway."
        })
    try:
        result = callback(action, params)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error("BotParlor tool error (%s): %s", action, e)
        return json.dumps({"error": str(e)})


def _get_manifest_url() -> str:
    """Resolve the BotParlor avatar manifest URL from config or env."""
    url = os.getenv("BOTPARLOR_MANIFEST_URL", "").strip()
    if url:
        return url
    try:
        import yaml
        from hermes_cli.config import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            bp_cfg = cfg.get("botparlor", {})
            if isinstance(bp_cfg, dict) and bp_cfg.get("manifest_url"):
                return bp_cfg["manifest_url"]
    except Exception:
        pass
    return "http://bothive.st-el.com:3000/api/avatar-manifest"


def list_moods_tool(args: dict, **kw) -> str:
    """Fetch available avatar moods from the BotParlor manifest."""
    bot_name = args.get("bot_name", "").strip().lower()

    try:
        url = _get_manifest_url()
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        manifest = resp.json()
    except Exception as e:
        logger.error("Failed to fetch avatar manifest: %s", e)
        return json.dumps({"error": f"Could not fetch avatar manifest: {e}"})

    if not bot_name:
        # Return list of bots with mood counts
        bots = {}
        for name, categories in manifest.items():
            if name == "avatar":
                continue
            moods = []
            for cat_moods in categories.values():
                moods.extend(cat_moods.keys())
            bots[name] = len(moods)
        return json.dumps({"bots": bots}, ensure_ascii=False)

    # Get moods for specific bot
    bot_data = manifest.get(bot_name)
    if not bot_data:
        available = [k for k in manifest.keys() if k != "avatar"]
        return json.dumps({
            "error": f"No avatar data for '{bot_name}'",
            "available_bots": available,
        })

    moods = []
    for category, cat_moods in bot_data.items():
        for mood_name, mood_info in cat_moods.items():
            entry = {
                "mood": mood_name,
                "category": category,
                "description": mood_info.get("description", ""),
            }
            # Include file type info (image vs video)
            files = mood_info.get("files", {})
            public = files.get("public", [])
            if public:
                ext = public[0].rsplit(".", 1)[-1].lower() if "." in public[0] else ""
                entry["type"] = "video" if ext in ("mp4", "webm") else "image"
            moods.append(entry)

    return json.dumps({
        "bot": bot_name,
        "mood_count": len(moods),
        "moods": moods,
    }, ensure_ascii=False)


def set_mood_tool(args: dict, **kw) -> str:
    """Change the bot's avatar mood/emotion."""
    mood = args.get("mood", "").strip()
    if not mood:
        return json.dumps({"error": "mood parameter is required"})
    return _botparlor_call("set_mood", {"mood": mood}, kw.get("callback"))


def display_media_tool(args: dict, **kw) -> str:
    """Display media content in the client's media panel."""
    url = args.get("url", "").strip()
    if not url:
        return json.dumps({"error": "url parameter is required"})
    return _botparlor_call("display_media", {
        "url": url,
        "type": args.get("type", "image"),
        "title": args.get("title", ""),
        "mode": args.get("mode", "panel"),
        "panelId": args.get("panelId"),
        "force": args.get("force", False),
    }, kw.get("callback"))


def take_photo_tool(args: dict, **kw) -> str:
    """Request a photo from the client's camera."""
    return _botparlor_call("sensor_request", {
        "sensorType": "camera",
        "camera": args.get("camera", "back"),
        "maxWidth": args.get("maxWidth", 1280),
        "maxHeight": args.get("maxHeight", 960),
        "quality": args.get("quality", 0.8),
    }, kw.get("callback"))


def get_location_tool(args: dict, **kw) -> str:
    """Request GPS coordinates from the client."""
    return _botparlor_call("sensor_request", {
        "sensorType": "location",
        "highAccuracy": args.get("highAccuracy", True),
    }, kw.get("callback"))


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_botparlor_requirements() -> bool:
    """BotParlor tools are always registered but only functional when
    a BotParlor client is connected via the gateway adapter."""
    return True


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

LIST_MOODS_SCHEMA = {
    "name": "list_moods",
    "description": (
        "Fetch the list of available avatar moods from the BotParlor manifest. "
        "Call with bot_name to get all moods for a specific bot (with descriptions "
        "and media types). Call without bot_name to get a summary of all bots and "
        "their mood counts. Use this before set_mood to know what moods are available."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "bot_name": {
                "type": "string",
                "description": "Bot name to list moods for (e.g., 'viper', 'lexi'). "
                               "Omit to list all bots with mood counts.",
            },
        },
        "required": [],
    },
}

SET_MOOD_SCHEMA = {
    "name": "set_mood",
    "description": (
        "Change your avatar's displayed mood/emotion in the chat interface. "
        "Use list_moods first to see what moods are available for your bot. "
        "Change mood naturally based on conversation context, up to once per "
        "response. Unknown moods fall back to 'default'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mood": {
                "type": "string",
                "description": "The mood/emotion name to set (e.g., 'happy', 'thinking', 'playful').",
            },
        },
        "required": ["mood"],
    },
}

DISPLAY_MEDIA_SCHEMA = {
    "name": "display_media",
    "description": (
        "Display media content (images, videos, PDFs, web pages, YouTube) in "
        "the client's media panel. The content appears alongside the chat in a "
        "dedicated panel. Use this to show visual content to the user rather "
        "than just linking to it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the media to display.",
            },
            "type": {
                "type": "string",
                "enum": ["image", "video", "youtube", "pdf", "webpage", "video_stream"],
                "description": "Type of media content.",
            },
            "title": {
                "type": "string",
                "description": "Optional title to display with the media.",
            },
            "mode": {
                "type": "string",
                "enum": ["panel", "fullscreen", "inline"],
                "description": "Display mode. 'panel' shows in side panel (default), "
                               "'fullscreen' takes over the view, 'inline' embeds in chat.",
            },
            "panelId": {
                "type": "integer",
                "description": "Target panel number (1, 2, etc.). Auto-selected if omitted.",
            },
            "force": {
                "type": "boolean",
                "description": "If true, replace existing content in the target panel.",
            },
        },
        "required": ["url", "type"],
    },
}

TAKE_PHOTO_SCHEMA = {
    "name": "take_photo",
    "description": (
        "Request a photo from the user's device camera. The photo is captured "
        "by the connected client (mobile app or web browser) and returned as "
        "a URL you can reference. Use 'front' camera for selfies, 'back' for "
        "general photos. This requires the user's camera permission and a "
        "connected client with camera capability."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "camera": {
                "type": "string",
                "enum": ["front", "back"],
                "description": "Which camera to use. 'front' for selfies, 'back' for general photos.",
            },
        },
        "required": [],
    },
}

GET_LOCATION_SCHEMA = {
    "name": "get_location",
    "description": (
        "Request the user's current GPS coordinates from their device. "
        "Returns latitude, longitude, and accuracy. Requires location "
        "permission on the connected client."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "highAccuracy": {
                "type": "boolean",
                "description": "Request high-accuracy GPS (slower but more precise). Default: true.",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="list_moods",
    toolset="botparlor",
    schema=LIST_MOODS_SCHEMA,
    handler=list_moods_tool,
    check_fn=check_botparlor_requirements,
)

registry.register(
    name="set_mood",
    toolset="botparlor",
    schema=SET_MOOD_SCHEMA,
    handler=set_mood_tool,
    check_fn=check_botparlor_requirements,
)

registry.register(
    name="display_media",
    toolset="botparlor",
    schema=DISPLAY_MEDIA_SCHEMA,
    handler=display_media_tool,
    check_fn=check_botparlor_requirements,
)

registry.register(
    name="take_photo",
    toolset="botparlor",
    schema=TAKE_PHOTO_SCHEMA,
    handler=take_photo_tool,
    check_fn=check_botparlor_requirements,
)

registry.register(
    name="get_location",
    toolset="botparlor",
    schema=GET_LOCATION_SCHEMA,
    handler=get_location_tool,
    check_fn=check_botparlor_requirements,
)
