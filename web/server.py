import asyncio
import json
import queue
import sys
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from agent.init_agents import DEFAULT_API_PROVIDERS, DEFAULT_LOCAL_MODELS
from agent.llm_factory import LOCAL_MODELS, PROVIDERS, get_api_key, provider_status
from engine.simulation_loop import run

app = FastAPI(title="LLM Arena")
_run_lock = threading.Lock()
_stop_event = threading.Event()

NATION_SLUGS = {
    "germany": "Germany",
    "france": "France",
    "uk": "United Kingdom",
}


def _error_stream(message):
    async def stream():
        payload = json.dumps({"type": "error", "message": message}, ensure_ascii=False)
        yield f"data: {payload}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def _build_model_config(params):
    config = {}
    for slug, nation in NATION_SLUGS.items():
        provider = (params.get(slug) or "ollama").strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}' for {nation}")
        if provider != "ollama" and get_api_key(provider) is None:
            raise ValueError(
                f"Missing API key for {nation}'s provider '{provider}'. "
                f"Set one of: {', '.join(PROVIDERS[provider]['env'])}"
            )
        model = (params.get(f"{slug}_model") or "").strip()
        if not model:
            model = (
                DEFAULT_LOCAL_MODELS[nation]
                if provider == "ollama"
                else PROVIDERS[provider]["default_model"]
            )
        config[nation] = {"provider": provider, "model": model}
    return config


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/providers")
def providers():
    return {
        "providers": provider_status(),
        "local_models": LOCAL_MODELS,
        "defaults": {
            "local": DEFAULT_LOCAL_MODELS,
            "api": DEFAULT_API_PROVIDERS,
        },
    }


@app.get("/api/status")
def status():
    acquired = _run_lock.acquire(blocking=False)
    if acquired:
        _run_lock.release()
        return {"running": False, "stopping": False}
    return {"running": True, "stopping": _stop_event.is_set()}


@app.get("/api/stop")
def stop():
    if _run_lock.locked():
        _stop_event.set()
        return {"stopping": True}
    return {"stopping": False}


@app.get("/api/run")
async def run_simulation(request: Request):
    params = dict(request.query_params)
    try:
        max_turns = int(params.get("max_turns", "10"))
    except ValueError:
        return _error_stream("max_turns must be an integer")
    max_turns = max(1, min(max_turns, 100))

    try:
        model_config = _build_model_config(params)
    except ValueError as exc:
        return _error_stream(str(exc))

    if not _run_lock.acquire(blocking=False):
        return _error_stream("A simulation is already running.")

    _stop_event.clear()
    events: "queue.Queue" = queue.Queue()

    def worker():
        try:
            run(
                max_turns=max_turns,
                verbose=False,
                on_event=lambda event: events.put(event),
                model_config=model_config,
                should_stop=_stop_event.is_set,
            )
        except Exception as exc:
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put(None)
            _run_lock.release()

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream():
        while True:
            event = await asyncio.to_thread(events.get)
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
