"""Streaming /v1/chat/completions must surface final_response when no content delta
was streamed (guardrail halt / partial_stream_recovery paths, #31449)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch


def _run_stream(queued, final_response):
    from aiohttp import web
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, ThreadSafeAsyncQueue

    adapter = APIServerAdapter(PlatformConfig(enabled=True, token="test-key"))
    written = []

    async def fake_agent():
        return {"final_response": final_response, "completed": True}, {
            "input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    async def run():
        stream_q = ThreadSafeAsyncQueue()
        for item in queued:
            stream_q.put_nowait(item)
        stream_q.put_nowait(None)
        agent_task = asyncio.ensure_future(fake_agent())
        resp = AsyncMock(spec=web.StreamResponse)
        resp.write = AsyncMock(side_effect=lambda data: written.append(data))
        resp.prepare = AsyncMock()
        req = MagicMock()
        req.headers = {}
        with patch("gateway.platforms.api_server.web.StreamResponse", return_value=resp):
            await adapter._write_sse_chat_completion(req, "cmpl-1", "m", 1, stream_q, agent_task)

    asyncio.run(run())
    contents = []
    for frame in written:
        for line in frame.decode().splitlines():
            if line.startswith("data: {"):
                delta = json.loads(line[6:])["choices"][0]["delta"]
                if delta.get("content"):
                    contents.append(delta["content"])
    return contents


def test_final_response_emitted_when_no_deltas_streamed():
    # No deltas at all: content_sent stays False and the recovery fallback
    # emits the final_response once. The data-URL filter is a no-op here
    # (nothing queued), so the single chunk is the whole response.
    assert _run_stream([], "recovered answer") == ["recovered answer"]


def test_final_response_not_duplicated_after_streamed_deltas():
    # Deltas came through, so the recovery fallback must NOT re-emit the
    # final_response (that would duplicate it). Assert the reassembled
    # assistant content equals the streamed text exactly, and that it
    # appears exactly once.
    #
    # Note: the stream now routes plain text through the data-URL filter,
    # which may hold back trailing characters that could begin a ``data:``
    # prefix (e.g. "world" -> "worl" + "d"). Chunk boundaries are therefore
    # not guaranteed to match the queued deltas byte-for-byte; the client-
    # visible content is, which is the contract this test pins.
    contents = _run_stream(["hello ", "world"], "hello world")
    reassembled = "".join(contents)
    assert reassembled == "hello world"
    assert reassembled.count("hello world") == 1
