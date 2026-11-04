"""Outbound media/document coverage for the API server: ``/v1/files`` exports, the inline
``data:`` URL rewriter (batch and SSE), and the multimodal response shapes that carry them.

Split out of ``test_api_server.py`` (its line cap); shared app/adapter helpers live there.
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import MAX_REQUEST_BYTES, _StreamDataURLFilter, _choose_export_ext
from tests.gateway.api_server_harness import create_app as _create_app, make_adapter as _make_adapter


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


class TestRequestBodyLimits:
    @pytest.mark.asyncio
    async def test_rejects_content_length_over_32mb(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                data=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(MAX_REQUEST_BYTES + 1),
                },
            )
            assert resp.status == 413

    @pytest.mark.asyncio
    async def test_app_sets_client_max_size_to_32mb(self, adapter):
        app = _create_app(adapter)
        assert app._client_max_size == MAX_REQUEST_BYTES

class TestChooseExportExt:
    """Covers the helper that picks a URL-safe extension for outbound
    /v1/files/{id}.ext URLs."""

    def test_prefers_on_disk_suffix(self, tmp_path):
        p = tmp_path / "voice.ogg"
        p.write_bytes(b"OggS\x00\x00")
        # Even though audio/ogg → mimetypes might return .ogx or similar on
        # some platforms, the real file extension wins.
        assert _choose_export_ext(p, "audio/ogg") == ".ogg"

    def test_lowercases_extension(self, tmp_path):
        p = tmp_path / "PHOTO.JPG"
        p.write_bytes(b"\xff\xd8\xff")
        assert _choose_export_ext(p, "image/jpeg") == ".jpg"

    def test_falls_back_to_mime_guess_when_no_suffix(self, tmp_path):
        p = tmp_path / "suffixless"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        # mimetypes.guess_extension("image/png") is ".png" on all
        # platforms shipped with Python 3.12.
        assert _choose_export_ext(p, "image/png") == ".png"

    def test_returns_empty_when_suffix_looks_unsafe(self, tmp_path):
        # A suffix that's too long / has weird chars is dropped rather
        # than propagated into the URL.  We fall through to the MIME
        # table, and if that also fails we return "".
        p = tmp_path / "thing.thisextensionisfartoolong"
        p.write_bytes(b"x")
        assert _choose_export_ext(p, "application/x-nonexistent-zzz") == ""

    def test_empty_when_nothing_works(self, tmp_path):
        p = tmp_path / "bare"
        p.write_bytes(b"x")
        # Unknown MIME → no guess → never invent an extension.
        assert _choose_export_ext(p, "application/x-nonexistent-zzz") == ""


class TestStreamDataURLFilter:
    """Direct unit tests for the streaming filter state machine.

    Uses a fake adapter whose ``_register_file_export`` returns a stable
    URL based on the cached-file path, so we can assert rewrites without
    spinning up an aiohttp Request.  End-to-end SSE coverage lives in
    ``TestChatCompletionsEndpoint.test_streaming_rewrites_inline_data_url``
    below.
    """

    class _FakeAdapter:
        def __init__(self):
            self.registered = []

        def _register_file_export(self, request, local_path):
            self.registered.append(local_path)
            # Mimic the real URL shape so tests can grep for /v1/files/.
            return f"https://host/v1/files/{Path(local_path).stem}{Path(local_path).suffix}"

    def _make_filter(self):
        adapter = self._FakeAdapter()
        return _StreamDataURLFilter(adapter, request=None), adapter

    def _build_png_data_url(self) -> str:
        import base64 as _b64
        raw = b"\x89PNG\r\n\x1a\nfake-png"
        return f"data:image/png;base64,{_b64.b64encode(raw).decode()}"

    def test_passes_through_plain_text(self):
        """No data URL, no change (modulo the tiny tail-hold that gets
        released on flush)."""
        f, _ = self._make_filter()
        emitted = f.feed("hello, world — nothing to see here")
        emitted += f.flush()
        assert emitted == "hello, world — nothing to see here"

    def test_rewrites_single_chunk_data_url(self):
        f, adapter = self._make_filter()
        data_url = self._build_png_data_url()
        text = f"Here: ![icon]({data_url})"
        emitted = f.feed(text) + f.flush()
        # No data URL remains and a /v1/files/ URL is spliced in.
        assert "data:" not in emitted, emitted
        assert "/v1/files/" in emitted, emitted
        # Markdown structure preserved.
        assert "![icon](" in emitted
        # One registration, one inline block recorded.
        assert len(adapter.registered) == 1
        assert len(f.inline_blocks) == 1
        assert f.inline_blocks[0]["type"] == "output_image"

    def test_rewrites_data_url_split_across_many_chunks(self):
        """Claude's tokens don't respect URL boundaries — ``data:`` may
        be split into pieces like ``"da"``/``"ta:image/png;base64,"``/
        ``"iVBORw0K"``/``"Ggo="``/``")"``."""
        f, adapter = self._make_filter()
        data_url = self._build_png_data_url()
        prefix = "Here: ![icon]("
        # Split the full text (prefix + URL + ")") into aggressively small
        # chunks — 3 chars each — to exercise every state transition.
        full = prefix + data_url + ")"
        chunks = [full[i:i + 3] for i in range(0, len(full), 3)]
        emitted_parts = [f.feed(ch) for ch in chunks]
        emitted_parts.append(f.flush())
        emitted = "".join(emitted_parts)
        assert "data:" not in emitted, emitted
        assert "/v1/files/" in emitted, emitted
        assert "![icon](" in emitted
        assert len(adapter.registered) == 1

    def test_data_url_terminator_is_markdown_close_paren(self):
        """The closing paren of ``![alt](…)`` must end the URL scan so
        anything after ``)`` is emitted as normal text."""
        f, _ = self._make_filter()
        data_url = self._build_png_data_url()
        emitted = f.feed(f"before ![x]({data_url}) after text") + f.flush()
        assert emitted.endswith(") after text"), emitted
        assert "/v1/files/" in emitted
        assert "data:" not in emitted

    def test_multiple_data_urls_each_materialized_once(self):
        f, adapter = self._make_filter()
        url1 = self._build_png_data_url()
        # Different payload → different cached file.
        import base64 as _b64
        raw2 = b"\x89PNG\r\n\x1a\nother-png"
        url2 = f"data:image/png;base64,{_b64.b64encode(raw2).decode()}"
        emitted = f.feed(f"![a]({url1}) and ![b]({url2})") + f.flush()
        assert emitted.count("/v1/files/") == 2, emitted
        assert "data:" not in emitted
        assert len(adapter.registered) == 2

    def test_undecodable_data_url_left_verbatim(self):
        """If ``_decode_data_url`` can't parse the payload, keep the URL in
        place — better to show the user a broken link than to silently eat
        bytes."""
        f, _ = self._make_filter()
        bad = "data:image/png;base64,@@@nope@@@"
        emitted = f.feed(f"Look: ![x]({bad})") + f.flush()
        # Original URL is preserved verbatim — filter doesn't invent one.
        assert bad in emitted, emitted

    def test_partial_data_prefix_at_buffer_boundary_is_held(self):
        """If a chunk ends with ``"da"`` and the next begins with
        ``"ta:image/png;base64,…"``, the filter must not emit the ``"da"``
        early — otherwise the ``data:`` match is lost."""
        f, _adapter = self._make_filter()
        data_url = self._build_png_data_url()
        first = f.feed("before da")  # last 2 chars are a partial prefix
        second = f.feed("ta:image/png;base64,")
        third = f.feed(data_url.split(",", 1)[1])  # payload
        fourth = f.feed(")")
        emitted = first + second + third + fourth + f.flush()
        assert "data:" not in emitted, emitted
        assert "/v1/files/" in emitted
        # "before " is stable enough that the first chunk may emit part
        # of it; by end-of-stream the full non-URL prose is present.
        assert emitted.startswith("before "), emitted

    def test_flush_handles_in_progress_data_url(self):
        """Stream ends mid-URL (no terminator reached) — the filter
        should still decode whatever payload bytes are valid rather than
        drop them."""
        f, _ = self._make_filter()
        data_url = self._build_png_data_url()
        # Deliberately *don't* include a closing paren / whitespace.
        emitted = f.feed(f"incomplete: {data_url}")
        emitted += f.flush()
        assert "data:" not in emitted, emitted
        assert "/v1/files/" in emitted

    def test_tool_progress_tuples_ignored_by_filter(self):
        """The SSE writer only sends *strings* through ``feed``; tool-
        progress events take a different path.  This is documented
        behaviour — verify the filter is string-only."""
        f, _ = self._make_filter()
        # Non-string input is a programming error; the caller in
        # _write_sse_chat_completion checks ``isinstance(item, str)``
        # before feeding.  Here we just confirm that empty chunks produce
        # empty output and don't wedge the state machine.
        assert f.feed("") == ""
        assert f.feed(None) == ""
        assert f.flush() == ""


# ---------------------------------------------------------------------------


class TestStreamingMediaTail:
    @pytest.mark.asyncio
    async def test_stream_appends_media_markdown_for_openwebui(self, adapter, tmp_path):
        image_path = tmp_path / "stream.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Done.")
                return (
                    {"final_response": f"Done.\nMEDIA:{image_path}", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "stream"}], "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()
                assert "Done." in body
                assert "/v1/files/" in body
                assert "![stream.png]" in body

    @staticmethod
    def _reassemble_sse_content(body: str) -> str:
        """Concatenate the ``delta.content`` of every SSE data frame in
        ``body`` into a single string — mirrors how an OpenAI client
        assembles a streaming reply into the assistant message."""
        parts: list[str] = []
        for line in body.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
        return "".join(parts)

    @pytest.mark.asyncio
    async def test_streaming_rewrites_inline_data_url(self, adapter):
        """End-to-end: Claude emits deltas that compose
        ``![icon](data:image/png;base64,…)`` across several chunks, and
        the text the client reassembles from the SSE stream must contain
        a ``/v1/files/`` URL instead of the base64 blob."""
        import base64 as _b64
        raw = b"\x89PNG\r\n\x1a\nfake-png-bytes"
        data_url = f"data:image/png;base64,{_b64.b64encode(raw).decode()}"
        # Break the text into chunks that *intentionally* split the
        # ``data:`` prefix across two deltas (last 2 chars of one chunk,
        # rest in the next) so the filter's prefix-hold path is exercised.
        prefix = "Here is your icon: ![icon]("
        deltas = [
            prefix[:-5],                         # "…: ![icon"
            prefix[-5:] + data_url[:2],          # "n](da"
            data_url[2:10],                      # "ta:image"
            data_url[10:],                       # remainder of the URL
            ")",                                 # markdown terminator
            "\n\nLet me know if you want changes.",  # trailing prose
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    for d in deltas:
                        cb(d)
                return (
                    {"final_response": prefix + data_url + ")" + "\n\nLet me know if you want changes.",
                     "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "icon"}], "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()
                reassembled = self._reassemble_sse_content(body)
                # No base64 payload reached the client…
                assert "data:image/" not in reassembled, reassembled
                assert "base64," not in reassembled, reassembled
                # …but the markdown wrapper and a /v1/files/ URL did.
                assert "![icon](" in reassembled
                assert re.search(r"/v1/files/[A-Za-z0-9]+\.png", reassembled), reassembled
                # Trailing prose that came AFTER the data URL still ships.
                assert "Let me know if you want changes." in reassembled

    @pytest.mark.asyncio
    async def test_streaming_plain_text_is_not_altered(self, adapter):
        """The filter's prefix-hold must not drop or reorder characters —
        after reassembly the stream matches the source tokens exactly."""
        tokens = ["Hello", ", ", "world", "! No ", "data here."]
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    for t in tokens:
                        cb(t)
                return (
                    {"final_response": "".join(tokens), "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()
                reassembled = self._reassemble_sse_content(body)
                assert reassembled == "Hello, world! No data here."
                # "data" (no colon) is plain text and must pass through
                # verbatim — the filter only reacts to the full ``data:``.
                assert "data here." in reassembled

    @pytest.mark.asyncio
    async def test_streaming_muted_turn_omits_terminal_media_tail(self, adapter, tmp_path):
        """A muted turn's terminal media tail must not ship: the merge-resolution guard
        for gating ``media_markdown`` on ``_notification_presentation_suppressed``."""
        image_path = tmp_path / "stream-secret.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("working on it")
                return (
                    {"final_response": f"Done.\nMEDIA:{image_path}", "messages": [], "api_calls": 1,
                     "_notification_presentation_suppressed": True},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "internal"}], "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()
                reassembled = self._reassemble_sse_content(body)
                assert "/v1/files/" not in reassembled
                assert "stream-secret" not in reassembled
                assert "![" not in reassembled

    @pytest.mark.asyncio
    async def test_streaming_recovery_path_without_content_delta_ships_final(self, adapter):
        """A recovery turn (guardrail halt / partial_stream_recovery) returns a
        final_response without firing any content delta. The fallback block must
        emit it — and must do so via the ``_resolve_media_to_data_urls`` name the
        method imports locally. A merge that dropped that import made this path
        raise NameError mid-stream."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                # No stream_delta_callback calls at all: content_sent stays False.
                return (
                    {"final_response": "Recovered summary.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()
                reassembled = self._reassemble_sse_content(body)
                assert "Recovered summary." in reassembled
                assert "data: [DONE]" in body


class TestChatCompletionsMedia:
    @pytest.mark.asyncio
    async def test_no_user_message_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "system", "content": "You are helpful."}],
                },
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_nonstream_media_response_includes_openwebui_markdown(self, adapter, tmp_path):
        image_path = tmp_path / "demo.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        mock_result = {
            "final_response": f"Generated.\nMEDIA:{image_path}",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "make image"}]},
                )
            assert resp.status == 200
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
            assert "MEDIA:" not in text
            assert "![" in text and "/v1/files/" in text
            assert "data:image/" not in text

    @pytest.mark.asyncio
    async def test_successful_completion(self, adapter):
        """Test a successful chat completion with mocked agent."""
        mock_result = {
            "final_response": "Hello! How can I help you today?",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "chat.completion"
            assert data["id"].startswith("chatcmpl-")
            assert data["model"] == "hermes-agent"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
            assert data["choices"][0]["message"]["content_parts"][0]["type"] == "output_text"
            assert data["choices"][0]["finish_reason"] == "stop"
            assert "usage" in data

    @pytest.mark.asyncio
    async def test_presentation_muted_omits_content_and_content_parts(self, adapter, tmp_path):
        """A notification-internal turn (_notification_presentation_suppressed) must leak
        neither text nor media: ``content`` is "" and the multimodal ``content_parts``
        extra is omitted entirely — the merge-resolution regression guard."""
        image_path = tmp_path / "secret.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        mock_result = {
            "final_response": f"Internal note.\nMEDIA:{image_path}",
            "messages": [],
            "api_calls": 1,
            "_notification_presentation_suppressed": True,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "internal"}]},
                )
            assert resp.status == 200
            data = await resp.json()
            message = data["choices"][0]["message"]
            assert message["content"] == ""
            assert "content_parts" not in message

    @pytest.mark.asyncio
    async def test_system_prompt_extracted(self, adapter):
        """System messages from the client are passed as ephemeral_system_prompt."""
        mock_result = {
            "final_response": "I am a pirate! Arrr!",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "system", "content": "You are a pirate."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )

            assert resp.status == 200
            # Check that _run_agent was called with the system prompt
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs.get("ephemeral_system_prompt") == "You are a pirate."
            assert call_kwargs.kwargs.get("user_message") == "Hello"

    @pytest.mark.asyncio
    async def test_conversation_history_passed(self, adapter):
        """Previous user/assistant messages become conversation_history."""
        mock_result = {"final_response": "3", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "1+1=?"},
                            {"role": "assistant", "content": "2"},
                            {"role": "user", "content": "Now add 1 more"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["user_message"] == "Now add 1 more"
            assert len(call_kwargs["conversation_history"]) == 2
            assert call_kwargs["conversation_history"][0] == {"role": "user", "content": "1+1=?"}
            assert call_kwargs["conversation_history"][1] == {"role": "assistant", "content": "2"}

    @pytest.mark.asyncio
    async def test_agent_error_returns_500(self, adapter):
        """Agent exception returns 500."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = RuntimeError("Provider failed")
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

            assert resp.status == 500
            data = await resp.json()
            assert "Provider failed" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_stable_session_id_across_turns(self, adapter):
        """Same conversation (same first user message) produces the same session_id."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        session_ids = []
        async with TestClient(TestServer(app)) as cli:
            # Turn 1: single user message
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )
                session_ids.append(mock_run.call_args.kwargs["session_id"])

            # Turn 2: same first message, conversation grew
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there!"},
                            {"role": "user", "content": "How are you?"},
                        ],
                    },
                )
                session_ids.append(mock_run.call_args.kwargs["session_id"])

        assert session_ids[0] == session_ids[1], "Session ID should be stable across turns"
        assert session_ids[0].startswith("api-"), "Derived session IDs should have api- prefix"

    @pytest.mark.asyncio
    async def test_different_conversations_get_different_session_ids(self, adapter):
        """Different first messages produce different session_ids."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        session_ids = []
        async with TestClient(TestServer(app)) as cli:
            for first_msg in ["Hello", "Goodbye"]:
                with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                    await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-agent",
                            "messages": [{"role": "user", "content": first_msg}],
                        },
                    )
                    session_ids.append(mock_run.call_args.kwargs["session_id"])

        assert session_ids[0] != session_ids[1]


# ---------------------------------------------------------------------------


class TestResponsesMediaOutput:
    @pytest.mark.asyncio
    async def test_response_output_includes_media_blocks_and_file_download(self, adapter, tmp_path):
        audio_path = tmp_path / "voice.ogg"
        audio_path.write_bytes(b"OggS\x00\x00")
        mock_result = {
            "final_response": f"[[audio_as_voice]]\nMEDIA:{audio_path}",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post("/v1/responses", json={"input": "reply with audio"})
            assert resp.status == 200
            data = await resp.json()
            content_blocks = data["output"][0]["content"]
            audio_blocks = [b for b in content_blocks if b.get("type") == "output_audio"]
            assert audio_blocks
            # Non-image media is never inlined — it always resolves to a
            # /v1/files/{id}.ext URL, even when the payload is tiny.
            assert "data:audio/" not in content_blocks[0]["text"]
            assert not audio_blocks[0]["audio_url"].startswith("data:")
            # Extension must match the source file (voice.ogg → .ogg).
            parsed = urlparse(audio_blocks[0]["audio_url"])
            assert re.fullmatch(r"/v1/files/[A-Za-z0-9]+\.ogg", parsed.path), parsed.path

    @pytest.mark.asyncio
    async def test_response_output_small_images_also_use_file_endpoint(self, adapter, tmp_path):
        """Images of any size go through /v1/files/{id}.ext — there's no
        inline-data-URL fast-path anymore."""
        image_path = tmp_path / "tiny.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nsmall")
        mock_result = {
            "final_response": f"MEDIA:{image_path}",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post("/v1/responses", json={"input": "tiny image"})
            assert resp.status == 200
            data = await resp.json()
            content_blocks = data["output"][0]["content"]
            image_blocks = [b for b in content_blocks if b.get("type") == "output_image"]
            assert image_blocks
            assert not image_blocks[0]["image_url"].startswith("data:")
            parsed = urlparse(image_blocks[0]["image_url"])
            assert re.fullmatch(r"/v1/files/[A-Za-z0-9]+\.png", parsed.path), parsed.path
            # Markdown fallback in the text also uses the file URL, not a
            # base64 data URL.
            assert "![tiny.png](" in content_blocks[0]["text"]
            assert "/v1/files/" in content_blocks[0]["text"]
            assert "data:image/" not in content_blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_inline_data_url_image_rewritten_to_file_url(self, adapter):
        """When the model embeds ``![alt](data:image/png;base64,…)`` in its
        reply (as Claude does for shell-produced images without a MEDIA
        convention), the server decodes the payload, caches it, and swaps
        the data URL for a ``/v1/files/{id}.png`` URL in-place.  The client
        sees a normal markdown image referencing a cheap HTTP URL instead
        of carrying base64 through conversation history."""
        import base64 as _b64
        png_bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes"
        data_url = f"data:image/png;base64,{_b64.b64encode(png_bytes).decode()}"
        mock_result = {
            "final_response": f"Here's your icon: ![icon]({data_url})",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post("/v1/responses", json={"input": "icon please"})
            assert resp.status == 200
            data = await resp.json()
            content_blocks = data["output"][0]["content"]
            # The text markdown wrapper survives intact, but its URL is now
            # a /v1/files/ URL — no base64 left in the response.
            text = content_blocks[0]["text"]
            assert "![icon](" in text
            assert "data:image/" not in text
            assert "data:" not in text
            match = re.search(r"/v1/files/[A-Za-z0-9]+\.(?:png|jpg|jpeg)", text)
            assert match, f"no /v1/files URL spliced into text: {text!r}"
            # And there's a proper output_image block referencing the same URL.
            image_blocks = [b for b in content_blocks if b.get("type") == "output_image"]
            assert image_blocks
            assert image_blocks[0]["image_url"].endswith(match.group(0).rsplit("/", 1)[-1])

    @pytest.mark.asyncio
    async def test_inline_data_url_audio_rewritten_to_file_url(self, adapter):
        """Audio data URLs the model embeds get the same treatment."""
        import base64 as _b64
        ogg_bytes = b"OggS\x00\x00fake-audio"
        data_url = f"data:audio/ogg;base64,{_b64.b64encode(ogg_bytes).decode()}"
        mock_result = {
            "final_response": f"voice note: [audio]({data_url})",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post("/v1/responses", json={"input": "audio please"})
            assert resp.status == 200
            data = await resp.json()
            content_blocks = data["output"][0]["content"]
            text = content_blocks[0]["text"]
            assert "data:" not in text
            # mimetypes maps ``audio/ogg`` to ``.oga`` on most systems and
            # ``.ogg`` on some — either is a legitimate Ogg extension.
            assert re.search(r"/v1/files/[A-Za-z0-9]+\.(?:ogg|oga)", text), text
            audio_blocks = [b for b in content_blocks if b.get("type") == "output_audio"]
            assert audio_blocks
            assert "/v1/files/" in audio_blocks[0]["audio_url"]

    @pytest.mark.asyncio
    async def test_inline_data_url_unparseable_left_in_place(self, adapter):
        """Garbled data URLs that fail to decode are left in the text
        verbatim rather than dropped — we'd rather show the user a
        malformed URL than silently delete content."""
        mock_result = {
            # Truncated base64 header — _decode_data_url returns empty bytes,
            # which the helper treats as "not materializable, skip."
            "final_response": "Broken: ![x](data:image/png;base64,@@@nope@@@)",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post("/v1/responses", json={"input": "garbled"})
            assert resp.status == 200
            # The response completes fine; no crash.  We don't assert the
            # exact text shape — only that nothing 500s.

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("filename", "payload"),
        [
            ("sample.pdf", b"%PDF-1.4\nmini\n"),
            ("sample.zip", b"PK\x03\x04mini"),
        ],
    )
    async def test_response_output_document_media_served_via_file_endpoint(
        self, adapter, tmp_path, filename, payload,
    ):
        """Documents resolve to /v1/files/{id}.ext URLs, regardless of size —
        like every other media kind, since the inline-data-URL path was
        removed."""
        file_path = tmp_path / filename
        file_path.write_bytes(payload)
        mock_result = {
            "final_response": f"MEDIA:{file_path}",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post("/v1/responses", json={"input": f"inline {filename}"})
            assert resp.status == 200
            data = await resp.json()
            content_blocks = data["output"][0]["content"]
            file_blocks = [b for b in content_blocks if b.get("type") == "output_file"]
            assert file_blocks
            assert not file_blocks[0]["file_url"].startswith("data:")
            assert "data:" not in content_blocks[0]["text"]
            # The served URL carries the real on-disk extension.
            expected_ext = Path(filename).suffix.lower()
            parsed = urlparse(file_blocks[0]["file_url"])
            assert re.fullmatch(
                rf"/v1/files/[A-Za-z0-9]+{re.escape(expected_ext)}", parsed.path,
            ), parsed.path

    @pytest.mark.asyncio
    async def test_response_output_file_endpoint_uses_forced_host_and_port(self, adapter, tmp_path, monkeypatch):
        image_path = tmp_path / "forced.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        mock_result = {
            "final_response": f"MEDIA:{image_path}",
            "messages": [],
            "api_calls": 1,
        }
        monkeypatch.setenv("API_SERVER_FILE_RESPONSE_FORCE_HOST", "files.example.internal")
        monkeypatch.setenv("API_SERVER_FILE_RESPONSE_FORCE_PORT", "9443")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post("/v1/responses", json={"input": "large image"})
            assert resp.status == 200
            data = await resp.json()
            content_blocks = data["output"][0]["content"]
            image_blocks = [b for b in content_blocks if b.get("type") == "output_image"]
            assert image_blocks
            parsed = urlparse(image_blocks[0]["image_url"])
            assert parsed.hostname == "files.example.internal"
            assert parsed.port == 9443
            assert re.fullmatch(r"/v1/files/[A-Za-z0-9]+\.png", parsed.path), parsed.path

    @pytest.mark.asyncio
    async def test_response_output_file_endpoint_uses_forced_port_only(self, adapter, tmp_path, monkeypatch):
        image_path = tmp_path / "forced-port.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        mock_result = {
            "final_response": f"MEDIA:{image_path}",
            "messages": [],
            "api_calls": 1,
        }
        monkeypatch.setenv("API_SERVER_FILE_RESPONSE_FORCE_PORT", "9000")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post("/v1/responses", json={"input": "large image"})
            assert resp.status == 200
            data = await resp.json()
            content_blocks = data["output"][0]["content"]
            image_blocks = [b for b in content_blocks if b.get("type") == "output_image"]
            assert image_blocks
            parsed = urlparse(image_blocks[0]["image_url"])
            assert parsed.port == 9000
            assert re.fullmatch(r"/v1/files/[A-Za-z0-9]+\.png", parsed.path), parsed.path



class TestFileEndpointAccess:
    @pytest.mark.asyncio
    async def test_files_endpoint_allows_token_access_without_auth_header(self, auth_adapter, tmp_path):
        export = tmp_path / "sample.txt"
        export.write_text("hello", encoding="utf-8")
        auth_adapter._file_exports["tok123"] = {"path": str(export), "created_at": time.time()}

        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/files/tok123")
            assert resp.status == 200
            assert await resp.text() == "hello"

    @pytest.mark.asyncio
    async def test_files_endpoint_accepts_matching_extension(self, auth_adapter, tmp_path):
        """A URL whose extension matches the one recorded at registration
        time resolves the same file as the extensionless URL."""
        export = tmp_path / "voice.ogg"
        export.write_bytes(b"OggS\x00\x00")
        auth_adapter._file_exports["aabb01"] = {
            "path": str(export), "created_at": time.time(), "ext": ".ogg",
        }

        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/files/aabb01.ogg")
            assert resp.status == 200
            assert await resp.read() == b"OggS\x00\x00"

    @pytest.mark.asyncio
    async def test_files_endpoint_rejects_mismatched_extension(self, auth_adapter, tmp_path):
        """A URL that guesses a different extension than the recorded one
        returns 404 — an attacker who learns the file_id can't fish for the
        real container format."""
        export = tmp_path / "voice.ogg"
        export.write_bytes(b"OggS\x00\x00")
        auth_adapter._file_exports["aabb02"] = {
            "path": str(export), "created_at": time.time(), "ext": ".ogg",
        }

        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/files/aabb02.mp3")
            assert resp.status == 404


# ---------------------------------------------------------------------------
