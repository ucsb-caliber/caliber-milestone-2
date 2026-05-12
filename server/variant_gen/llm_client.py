"""OpenRouter chat completions; JSON in/out for generate + verify steps."""

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import SSLError

from .config import (
    BASE_TEMPERATURE,
    DEBUG,
    OPENROUTER_URL,
    openrouter_timeout_generate,
    openrouter_vision_enabled,
    resolved_openrouter_model,
)

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def encode_image(path: Union[Path, str], data_url: bool = False) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    if not data_url:
        return b64
    mime = _IMAGE_MIME.get(p.suffix.lower(), "image/png")
    return f"data:{mime};base64,{b64}"


def parse_llm_json(content: Optional[str]) -> Optional[Dict[str, Any]]:
    if not content or not isinstance(content, str):
        return None
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"  Model returned invalid JSON: {e}")
        if DEBUG:
            print(f"[DEBUG] Raw content was: {content[:2000]}")
        return None


def _openrouter_chat(
    prompt: str,
    image_paths: List[Path],
    temp: float,
    model: str,
    timeout_sec: float,
) -> Optional[Dict[str, Any]]:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("  OPENROUTER_API_KEY is not set.")
        return None

    if image_paths:
        parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for p in image_paths:
            url = encode_image(p, data_url=True)
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
        user_content: Any = parts
    else:
        user_content = prompt

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    ref = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    if ref:
        headers["HTTP-Referer"] = ref
    title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
    if title:
        headers["X-Title"] = title

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temp,
        "response_format": {"type": "json_object"},
    }
    connect_s = min(20.0, max(5.0, timeout_sec * 0.25))
    max_attempts = max(1, int(os.getenv("OPENROUTER_HTTP_RETRIES", "3")))
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                OPENROUTER_URL,
                json=payload,
                headers=headers,
                timeout=(connect_s, timeout_sec),
            )
            if not response.ok:
                try:
                    err = response.json()
                    detail = err.get("error", err)
                except Exception:
                    detail = response.text[:500]
                print(f"  OpenRouter HTTP {response.status_code}: {detail}")
                return None
            data = response.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if DEBUG:
                print(f"[DEBUG] Raw response: {content}")
            return parse_llm_json(content)
        except requests.Timeout:
            print(f"  Request timed out ({timeout_sec:.0f}s).")
            return None
        except (SSLError, RequestsConnectionError) as e:
            if attempt + 1 >= max_attempts:
                print(f"  Unexpected error: {type(e).__name__}: {e}")
                return None
            delay = min(8.0, 0.6 * (2**attempt))
            print(
                f"  Transient network/TLS error ({type(e).__name__}), "
                f"retry {attempt + 2}/{max_attempts} in {delay:.1f}s..."
            )
            time.sleep(delay)
        except Exception as e:
            print(f"  Unexpected error: {type(e).__name__}: {e}")
            return None
    return None


def call_llm(
    prompt: str,
    *,
    image_paths: Optional[List[Union[Path, str]]] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Generate or verify: JSON object in, JSON object out. Uses OpenRouter only."""
    paths = [Path(x) for x in (image_paths or []) if x]
    temp = temperature if temperature is not None else BASE_TEMPERATURE
    mid = model or resolved_openrouter_model()
    to = float(timeout_sec) if timeout_sec is not None else openrouter_timeout_generate()
    if DEBUG:
        print(f"\n[DEBUG] Calling OpenRouter model: {mid} (timeout {to:.0f}s)")
    return _openrouter_chat(prompt, paths, temp, mid, to)


def text_model_supports_images() -> bool:
    return openrouter_vision_enabled()
