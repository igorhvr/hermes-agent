"""Outbound media/document export for the API server: inline ``data:`` URLs and ``MEDIA:`` tags
become ``/v1/files/{id}.ext`` URLs the client can download.

Split out of ``api_server.py`` (a facade over its line cap): the adapter owns no state of its own
here beyond the export registry ``_file_exports`` it initialises, so this is a plain mixin plus the
module-level helpers the chat-completions routes reuse.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote_to_bytes, urlsplit

from aiohttp import web

MAX_EXPORTED_FILES = 256
FILE_EXPORT_TTL_SECONDS = 14_400

# Recognises a "normal" file extension: leading dot, then 1-8 alphanumerics.
# Used to scrub both the on-disk suffix and ``mimetypes.guess_extension``
# output before exposing it in a public URL.
_SAFE_EXT_RE = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


# Matches ``data:<mime/attrs>,<payload>`` URLs the model embeds in its final
# reply -- almost always inside markdown like ``![alt](data:image/png;base64,...)``.
_INLINE_DATA_URL_RE = re.compile(
    r'data:[^\s,)"\'<>]+,[A-Za-z0-9+/=%._~\-]+',
    re.IGNORECASE,
)

# MIME top level -> (output block type, block URL key). Anything else is a plain file; the block
# shape is one contract, so the two producers (live stream, final reply) never drift.
_MEDIA_BLOCK_ROWS = (
    ("image", "output_image", "image_url"),
    ("audio", "output_audio", "audio_url"),
    ("video", "output_video", "video_url"),
)

# Block type -> (URL key, markdown template) for the text-only client fallback.
_MEDIA_MARKDOWN_ROWS = {
    "output_image": "![{name}]({url})",
    "output_audio": "[Audio: {name}]({url})",
    "output_video": "[Video: {name}]({url})",
    "output_file": "[File: {name}]({url})",
}

# Extension fallbacks for the reference scan, which classifies by name (a bare path has no MIME).
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
_AUDIO_EXTS = frozenset({".ogg", ".opus", ".mp3", ".wav", ".m4a"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


def _media_block(mime_primary: str, url: str, filename: str) -> dict[str, Any]:
    """One ``output_*`` content block for *url*, shaped by the MIME top level."""
    top = (mime_primary or "").split("/", 1)[0].strip().lower()
    for kind, block_type, url_key in _MEDIA_BLOCK_ROWS:
        if top == kind:
            return {"type": block_type, url_key: url, "mime_type": mime_primary, "filename": filename}
    return {"type": "output_file", "file_url": url, "mime_type": mime_primary, "filename": filename}


def _decode_data_url(data_url: str) -> tuple[str, bytes]:
    """Decode a data URL into ``(mime_type, raw_bytes)``."""
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ValueError("Not a data URL")
    header, sep, payload = data_url.partition(",")
    if not sep:
        raise ValueError("Malformed data URL")
    meta = header[5:]  # strip 'data:'
    mime = "application/octet-stream"
    if meta:
        mime = (meta.split(";", 1)[0] or mime).strip() or mime
    is_b64 = ";base64" in meta
    try:
        if is_b64:
            raw = base64.b64decode(payload, validate=False)
        else:
            raw = unquote_to_bytes(payload)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid data URL payload: {exc}") from exc
    return mime, raw


def _ext_for_mime(mime: str, fallback: str) -> str:
    guessed = mimetypes.guess_extension((mime or "").split(";", 1)[0].strip() or "")
    if guessed:
        return guessed
    return fallback


def _choose_export_ext(path: Path, mime: str) -> str:
    """Pick a URL-safe extension for an outbound file export.

    Preference order:

    1. The file's on-disk suffix, lowercased, if it matches ``_SAFE_EXT_RE``.
    2. ``mimetypes.guess_extension(mime)`` as a fallback when the file
       itself has no suffix (e.g. things cached with a bare uuid name).
    3. Empty string -- never invent ``.bin`` or similar, because a wrong
       extension is worse for clients sniffing by URL than none at all.
    """
    suffix = path.suffix.lower()
    if _SAFE_EXT_RE.match(suffix):
        return suffix
    guessed = mimetypes.guess_extension((mime or "").split(";", 1)[0].strip() or "")
    if guessed and _SAFE_EXT_RE.match(guessed):
        return guessed.lower()
    return ""


def _cache_data_url_bytes(mime_primary: str, raw: bytes) -> str:
    """Write decoded bytes into the media cache and return the cached path."""
    from gateway.platforms.base import cache_audio_from_bytes, cache_document_from_bytes, cache_image_from_bytes

    top = (mime_primary or "").split("/", 1)[0]
    if top == "image":
        return cache_image_from_bytes(raw, ext=_ext_for_mime(mime_primary, ".png"))
    if top == "audio":
        return cache_audio_from_bytes(raw, ext=_ext_for_mime(mime_primary, ".ogg"))
    ext = _ext_for_mime(mime_primary, ".bin")
    return cache_document_from_bytes(raw, f"inline_{uuid.uuid4().hex[:8]}{ext}")


class _StreamDataURLFilter:
    """Rewrite ``data:…`` URLs to ``/v1/files/…`` URLs in a streamed text.

    The SSE chat-completions path forwards every agent token to the client
    the instant it arrives from the LLM, which means a reply like
    ``![icon](data:image/png;base64,iVBORw0KG…)`` reaches the client as a
    base64 blob before the server ever sees the complete URL.  By the time
    the batch-path rewriter (_materialize_inline_data_urls) runs on the
    agent's ``final_response``, the stream has already shipped the raw
    payload and there's no amendment mechanism in the OpenAI SSE spec.

    This state machine sits between the agent's delta queue and the SSE
    writer.  It buffers incoming text, suppresses anything that looks like
    a ``data:`` URL from the downstream emitter, and on URL completion
    decodes the payload, caches the bytes under
    ``~/.hermes/{image,audio,document}_cache``, registers the cached path
    with the file-export registry, and emits the resulting
    ``/v1/files/{id}.ext`` URL in place of the original data URL.

    Failure modes are conservative: an undecodable payload, a cache-helper
    exception, or a failed registration leaves the original data URL
    string in the output rather than dropping user-visible content.
    """

    # Characters that end a data URL embedded in markdown / HTML.  Closing
    # paren for ``![alt](…)``, whitespace, quotes, angle brackets.
    _TERMINATORS = frozenset(' \t\n\r)"\'<>')
    # The prefix we're watching for.  Held-back tail length is bounded by
    # ``len(_PREFIX) - 1`` (4 chars) but only activates when the buffer
    # actually ends with a prefix of this string — see ``_prefix_holdback``.
    _PREFIX = "data:"

    def __init__(self, adapter, request):
        self._adapter = adapter
        self._request = request
        self._buffer = ""
        self._in_url = False
        self.inline_blocks: list = []

    def feed(self, chunk: str) -> str:
        """Absorb ``chunk`` into the buffer; return text safe to emit now."""
        if not chunk:
            return ""
        self._buffer += chunk
        return self._drain(force=False)

    def flush(self) -> str:
        """End of stream — decode any in-progress data URL and return
        everything still buffered."""
        return self._drain(force=True)

    def _prefix_holdback(self) -> int:
        """Return how many trailing characters to keep buffered because
        they could be the start of ``data:``.  Zero when the tail doesn't
        match any prefix of the target string — the common case for
        ordinary prose — so normal tokens pass through untouched."""
        buf = self._buffer
        max_k = min(len(self._PREFIX) - 1, len(buf))
        for k in range(max_k, 0, -1):
            if buf[-k:] == self._PREFIX[:k]:
                return k
        return 0

    def _drain(self, force: bool) -> str:
        out: list[str] = []
        while self._buffer:
            if self._in_url:
                term_idx = -1
                for i, ch in enumerate(self._buffer):
                    if ch in self._TERMINATORS:
                        term_idx = i
                        break
                if term_idx < 0:
                    if force:
                        data_url = self._buffer
                        self._buffer = ""
                        out.append(self._materialize(data_url))
                        self._in_url = False
                    else:
                        break
                else:
                    data_url = self._buffer[:term_idx]
                    self._buffer = self._buffer[term_idx:]
                    out.append(self._materialize(data_url))
                    self._in_url = False
            else:
                idx = self._buffer.find("data:")
                if idx >= 0:
                    if idx > 0:
                        out.append(self._buffer[:idx])
                    self._buffer = self._buffer[idx:]
                    self._in_url = True
                else:
                    if force:
                        out.append(self._buffer)
                        self._buffer = ""
                    else:
                        hold = self._prefix_holdback()
                        safe_len = len(self._buffer) - hold
                        if safe_len > 0:
                            out.append(self._buffer[:safe_len])
                            self._buffer = self._buffer[safe_len:]
                    break
        return "".join(out)

    def _materialize(self, data_url: str) -> str:
        """Decode one data URL and return its ``/v1/files/…`` replacement.

        On any failure return the original ``data_url`` verbatim — a broken
        URL in the output is strictly better than silently dropping bytes
        the model intended to send.
        """
        try:
            mime, raw = _decode_data_url(data_url)
        except ValueError:
            return data_url
        if not raw:
            return data_url
        mime_primary = (mime or "").split(";", 1)[0].strip().lower() or "application/octet-stream"
        try:
            cached = _cache_data_url_bytes(mime_primary, raw)
        except Exception:  # health: allow BLE001 -- cache helpers fail on disk errors / odd MIMEs; the raw URL stays valid
            return data_url
        url = self._adapter._register_file_export(self._request, cached)
        if not url:
            return data_url
        self.inline_blocks.append(_media_block(mime_primary, url, Path(cached).name))
        return url


class FileExportMixin:
    """``/v1/files/{id}.ext`` serving, the export registry, and the media rewriting the
    chat-completions/Responses routes apply to a final reply. Mixed into ``APIServerAdapter``,
    which initialises ``_file_exports``."""

    async def _handle_get_file(self, request: web.Request) -> web.Response:
        """GET /v1/files/{file_id}[.ext] — download an exported local media/document file.

        The ``.ext`` suffix is cosmetic (so clients sniffing by URL see the
        right container) but it must match the one recorded at registration
        time if supplied, otherwise we 404.  This stops anyone who guessed
        a file_id from probing for the real extension.
        """
        self._prune_file_exports()
        from gateway.platforms.api_server import _openai_error

        file_id = request.match_info["file_id"]
        ext_in_url = request.match_info.get("ext", "") or ""
        meta = self._file_exports.get(file_id)
        if not meta:
            return web.json_response(_openai_error("File not found", code="file_not_found"), status=404)
        expected_ext = str(meta.get("ext", "") or "")
        if ext_in_url and ext_in_url.lower() != expected_ext.lower():
            return web.json_response(_openai_error("File not found", code="file_not_found"), status=404)
        path = Path(str(meta.get("path", "")))
        if not await asyncio.to_thread(path.is_file):
            self._file_exports.pop(file_id, None)
            return web.json_response(_openai_error("File no longer available", code="file_not_found"), status=404)
        return web.FileResponse(path)

    # ------------------------------------------------------------------
    # File export (outbound media/document download)
    # ------------------------------------------------------------------

    def _prune_file_exports(self) -> None:
        now = time.time()
        expired = [k for k, v in self._file_exports.items() if (now - float(v.get("created_at", 0))) > FILE_EXPORT_TTL_SECONDS]
        for k in expired:
            self._file_exports.pop(k, None)
        if len(self._file_exports) > MAX_EXPORTED_FILES:
            oldest = sorted(self._file_exports.items(), key=lambda kv: float(kv[1].get("created_at", 0)))
            for key, _ in oldest[: len(self._file_exports) - MAX_EXPORTED_FILES]:
                self._file_exports.pop(key, None)

    def _register_file_export(self, request: web.Request, local_path: str) -> Optional[str]:
        try:
            p = Path(local_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            # resolve() only fails on a bad path (loop, over-long, missing home) — keep the reply going.
            return None
        if not p.is_file():
            return None
        self._prune_file_exports()
        file_id = uuid.uuid4().hex
        mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        ext = _choose_export_ext(p, mime)
        self._file_exports[file_id] = {"path": str(p), "created_at": time.time(), "ext": ext}
        authority = self._file_response_authority(request)
        return f"{request.scheme}://{authority}/v1/files/{file_id}{ext}"

    @staticmethod
    def _file_response_authority(request: web.Request) -> str:
        """The ``host[:port]`` an exported URL advertises, honouring the FORCE_HOST/PORT
        overrides a NAT or reverse-proxy deployment needs."""
        forced_host = (os.getenv("API_SERVER_FILE_RESPONSE_FORCE_HOST", "") or "").strip()
        forced_port = (os.getenv("API_SERVER_FILE_RESPONSE_FORCE_PORT", "") or "").strip().lstrip(":")
        if not (forced_host or forced_port):
            return request.host
        # Keep whichever half the operator did not override.
        current_host, current_port = request.host, ""
        parsed = urlsplit(f"//{request.host}")
        if parsed.hostname:
            current_host = parsed.hostname
        if parsed.port:
            current_port = str(parsed.port)
        host = forced_host or current_host
        port = forced_port or current_port
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"  # IPv6 literal
        return f"{host}:{port}" if port else host

    @staticmethod
    def _render_media_markdown(blocks: list[dict[str, Any]]) -> str:
        """Render media blocks into markdown that text-only clients can display."""
        lines: list[str] = []
        for block in blocks:
            template = _MEDIA_MARKDOWN_ROWS.get(str(block.get("type", "")))
            if template is None:
                continue
            url_key = str(block.get("type", "")).replace("output_", "") + "_url"
            url = str(block.get(url_key) or "").strip()
            if not url:
                continue
            name = str(block.get("filename", "")).strip() or "media"
            lines.append(template.format(name=name, url=url))
        return "\n".join(lines)

    def _register_materialized(self, request: web.Request, mime_primary: str, raw: bytes) -> Optional[tuple[str, dict[str, Any]]]:
        """Cache decoded bytes and register them for download -> ``(url, block)`` or None."""
        try:
            cached = _cache_data_url_bytes(mime_primary, raw)
        except Exception:  # health: allow BLE001 -- cache helpers fail on disk errors / odd MIMEs; the caller keeps the raw URL
            return None
        url = self._register_file_export(request, cached)
        if not url:
            return None
        return url, _media_block(mime_primary, url, Path(cached).name)

    def _materialize_inline_data_urls(
        self, request: web.Request, text: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Rewrite model-embedded ``data:`` URLs into ``/v1/files/`` URLs in-place.

        LLMs (Claude most notably) frequently return images produced by shell
        tools as ``![alt](data:image/png;base64,…)`` in their final reply.
        That inlined payload bypasses the ``MEDIA:<path>`` convention, bloats
        response JSON, and makes conversation history carry the bytes on
        every turn.  This helper intercepts those data URLs, decodes the
        payload, caches the bytes under ``~/.hermes/{image,audio,document}_cache``,
        registers the cached path with the file-export registry, and splices
        the resulting ``/v1/files/{id}.ext`` URL into the text in place of
        the original data URL.  The enclosing markdown structure
        (``![alt](…)``) is preserved verbatim — only the URL swaps.

        Returns ``(rewritten_text, blocks)`` where ``blocks`` is the list of
        spec-compliant ``output_*`` blocks corresponding to the materialised
        data URLs (these go into the content-parts array alongside any
        MEDIA-tag / bare-path blocks the caller discovers).  An empty list
        means the text held no decodable data URLs.
        """
        if not text or "data:" not in text:
            return text, []

        blocks: list[dict[str, Any]] = []
        materialized: dict[str, str] = {}  # original data URL → new URL

        for match in _INLINE_DATA_URL_RE.finditer(text):
            data_url = match.group(0)
            if data_url in materialized:
                continue
            try:
                mime, raw = _decode_data_url(data_url)
            except ValueError:
                continue
            if not raw:
                continue
            mime_primary = (mime or "").split(";", 1)[0].strip().lower() or "application/octet-stream"
            registered = self._register_materialized(request, mime_primary, raw)
            if registered is None:
                # Cache/registration failure: leave the original data URL rather than drop the bytes.
                continue
            materialized[data_url], block = registered
            blocks.append(block)

        if not materialized:
            return text, []
        # Longest-first pass so a shorter data URL can't accidentally match
        # inside a longer one that shares a prefix.
        rewritten = text
        for original, new_url in sorted(materialized.items(), key=lambda kv: -len(kv[0])):
            rewritten = rewritten.replace(original, new_url)
        return rewritten, blocks

    def _extract_outbound_media_parts(self, request: web.Request, final_text: str) -> tuple[str, list[dict[str, Any]], str]:
        from gateway.platforms.base import BasePlatformAdapter

        # First: turn any inline ``data:`` URLs the model wrote by hand into
        # ``/v1/files/`` URLs directly inside the markdown.  The returned
        # ``inline_blocks`` go straight into the content-parts array; they
        # must NOT be re-rendered by _render_media_markdown because the
        # markdown ``![alt](URL)`` the model wrote is already in the text —
        # only its URL needed swapping.
        final_text, inline_blocks = self._materialize_inline_data_urls(request, final_text or "")
        tagged, cleaned = BasePlatformAdapter.extract_media(final_text or "")
        bare_paths, cleaned = BasePlatformAdapter.extract_local_files(cleaned or "")

        refs: list[str] = [path for path, _ in tagged] + list(bare_paths)
        seen: set[str] = set()
        blocks: list[dict[str, Any]] = []
        for ref in refs:
            if not ref or ref in seen:
                continue
            seen.add(ref)
            mime, _ = mimetypes.guess_type(ref)
            mime = mime or "application/octet-stream"
            resolved_path = Path(ref).expanduser().resolve()
            filename = resolved_path.name
            # All outbound media — images included — is served from
            # /v1/files/{id}.ext.  Inlining as data URLs was a micro-opt for
            # tiny thumbnails that just bloated response JSON and broke
            # conversation-history round-trips.
            url = self._register_file_export(request, ref)
            if not url:
                continue
            ext = resolved_path.suffix.lower()
            blocks.append(_media_block(self._mime_for_reference(mime, ext), url, filename))
        # Markdown is the fallback for clients that render the ``text`` field
        # only — it must cover MEDIA-tag / bare-path blocks whose URLs aren't
        # already in the text, but NOT the inline-data-URL blocks whose
        # markdown ``![alt](URL)`` wrapper is still present in ``cleaned``.
        markdown = self._render_media_markdown(blocks)
        return cleaned, inline_blocks + blocks, markdown

    @staticmethod
    def _mime_for_reference(mime: str, ext: str) -> str:
        """Classify a bare media path whose guessed MIME is generic, by extension."""
        if mime != "application/octet-stream":
            return mime
        if ext in _IMAGE_EXTS:
            return f"image/{ext.lstrip('.')}"
        if ext in _AUDIO_EXTS:
            return "audio/x-media"
        if ext in _VIDEO_EXTS:
            return "video/x-media"
        return mime


def finalize_batch_reply(adapter, request: web.Request, final_response: str) -> tuple[str, list[dict[str, Any]]]:
    """``(text, media blocks)`` for a non-streaming chat reply: ``MEDIA:`` tags resolved to data
    URLs, those (and any the model wrote itself) rewritten to ``/v1/files`` URLs, and the media
    markdown appended to the text."""
    from gateway.platforms.api_server import _resolve_media_to_data_urls

    resolved = _resolve_media_to_data_urls(final_response or "")
    cleaned, blocks, markdown = adapter._extract_outbound_media_parts(request, resolved)
    text = cleaned or resolved
    if markdown:
        text = f"{text}\n\n{markdown}".strip()
    return text, blocks


def streamed_media_tail(adapter, request: web.Request, result: Any) -> str:
    """The media markdown tail for a finished streamed turn ("" when the reply carries none)."""
    if not isinstance(result, dict):
        return ""
    final_response = result.get("final_response", "") or result.get("error", "")
    if not final_response:
        return ""
    return adapter._extract_outbound_media_parts(request, final_response)[2]


class StreamTextEmitter:
    """``delta.content`` writer for a streamed reply: text goes through the data-URL filter, which
    may hold back a partial ``data:`` prefix, and its held-back tail is flushed at end of stream.
    ``content_sent`` records whether any content reached the client."""

    def __init__(self, adapter, request: web.Request, write_content):
        self._filter = _StreamDataURLFilter(adapter, request)
        self._write_content = write_content
        self.content_sent = False

    async def feed(self, text: str) -> None:
        safe = self._filter.feed(text)
        if safe:
            self.content_sent = True
            await self._write_content(safe)

    async def flush(self) -> None:
        tail = self._filter.flush()
        if tail:
            self.content_sent = True
            await self._write_content(tail)


def message_presentation_extras(
    muted: bool, final_text: str, media_blocks: list[dict[str, Any]], reasoning_text: str = "",
) -> dict[str, Any]:
    """Post-processing extras for the assistant message of a batch reply.

    Both ride the same gate: a notification-internal turn (presentation suppressed) leaks neither
    reasoning nor media. ``content_parts`` always carries the text block — media appends to it."""
    if muted:
        return {}
    extras: dict[str, Any] = {"content_parts": [{"type": "output_text", "text": final_text}] + media_blocks}
    if reasoning_text:
        extras["reasoning_content"] = reasoning_text
    return extras
