"""Shared adapter/app harness for the API server test modules.

Kept out of ``test_api_server.py`` (whose line cap the feature work pressed against): every route the
OpenAI-compatible surface exposes is registered here so a test module only chooses what to exercise.
"""

from aiohttp import web

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    MAX_REQUEST_BYTES,
    APIServerAdapter,
    body_limit_middleware,
    cors_middleware,
    security_headers_middleware,
)


def make_adapter(api_key: str = "", cors_origins=None) -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    if cors_origins is not None:
        extra["cors_origins"] = cors_origins
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def create_app(adapter: APIServerAdapter) -> web.Application:
    """Create the aiohttp app from the adapter (without starting the full server)."""
    mws = [mw for mw in (cors_middleware, body_limit_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws, client_max_size=MAX_REQUEST_BYTES)
    app["api_server_adapter"] = adapter
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/health/detailed", adapter._handle_health_detailed)
    app.router.add_get("/v1/health", adapter._handle_health)
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_get("/api/model/options", adapter._handle_model_options)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/v1/skills", adapter._handle_skills)
    app.router.add_get("/v1/toolsets", adapter._handle_toolsets)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    app.router.add_delete("/v1/responses/{response_id}", adapter._handle_delete_response)
    app.router.add_post(
        "/api/platforms/{platform}/events",
        adapter._handle_platform_event_callback,
    )
    app.router.add_get(
        r"/v1/files/{file_id:[A-Za-z0-9]+}{ext:(?:\.[A-Za-z0-9]{1,8})?}",
        adapter._handle_get_file,
    )
    return app
