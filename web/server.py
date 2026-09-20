import asyncio
import json
import queue
import sys
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from agent.init_agents import (
    DEFAULT_API_PROVIDERS,
    DEFAULT_LOCAL_MODELS,
    default_model_config,
)
from agent.llm_factory import (
    LOCAL_MODELS,
    PROVIDERS,
    build_chat_model,
    get_api_key,
    provider_status,
    report_env_config,
)
from engine.savegame import (
    delete_save,
    list_saves,
    load_game,
    new_save_id,
    write_save_payload,
)
from engine.simulation_loop import run

app = FastAPI(title="LLM Arena")
_run_lock = threading.Lock()
_stop_event = threading.Event()

# Latest clean turn-boundary snapshot of the running (or last) game, used by
# /api/save so we never serialise a half-finished turn. Guarded by _state_lock.
_latest_state = None
_state_lock = threading.Lock()
# The SSE queue of the active run, so out-of-band events (e.g. "saved") can be
# injected into the live feed.
_active_events: "queue.Queue | None" = None

# API keys entered in the Web UI. Kept in memory for this local server process;
# they take precedence over environment variables for the duration of a run.
_api_keys = {}
_api_keys_lock = threading.Lock()

# Optional override for the save directory (tests point this at a temp dir).
SAVE_DIR = None

NATION_SLUGS = {
    "germany": "Germany",
    "france": "France",
    "uk": "United Kingdom",
    "ussr": "Soviet Union",
}

AUTOSAVE_ID = "autosave"


def _error_stream(message):
    async def stream():
        payload = json.dumps({"type": "error", "message": message}, ensure_ascii=False)
        yield f"data: {payload}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def _build_model_config(params):
    config = {}
    env_defaults = default_model_config()
    with _api_keys_lock:
        overrides = dict(_api_keys)
    for slug, nation in NATION_SLUGS.items():
        env_conf = env_defaults.get(nation, {})
        # Explicit UI selection wins; otherwise fall back to the .env default.
        provider = (params.get(slug) or "").strip().lower()
        if not provider:
            provider = env_conf.get("provider", "ollama")
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}' for {nation}")
        entry = {"provider": provider}
        if provider != "ollama":
            api_key = get_api_key(provider, overrides.get(provider))
            if api_key is None:
                raise ValueError(
                    f"Missing API key for {nation}'s provider '{provider}'. "
                    f"Enter it in the Web UI or set one of: "
                    f"{', '.join(PROVIDERS[provider]['env'])}"
                )
            entry["api_key"] = api_key
        model = (params.get(f"{slug}_model") or "").strip()
        if not model and env_conf.get("provider") == provider:
            model = env_conf.get("model") or ""
        if not model:
            model = (
                DEFAULT_LOCAL_MODELS[nation]
                if provider == "ollama"
                else PROVIDERS[provider]["default_model"]
            )
        entry["model"] = model
        config[nation] = entry
    return config


def _resume_model_config(saved_models):
    """Rebuild a model config from a save, resolving API keys as usual."""
    saved_models = saved_models or {}
    with _api_keys_lock:
        overrides = dict(_api_keys)
    config = {}
    for nation in NATION_SLUGS.values():
        spec = saved_models.get(nation) or {}
        provider = (spec.get("provider") or "ollama").strip().lower()
        if provider not in PROVIDERS:
            provider = "ollama"
        entry = {"provider": provider}
        if provider != "ollama":
            api_key = get_api_key(provider, overrides.get(provider))
            if api_key is None:
                raise ValueError(
                    f"Missing API key for {nation}'s provider '{provider}'. "
                    f"Enter it in the Web UI or set one of: "
                    f"{', '.join(PROVIDERS[provider]['env'])}"
                )
            entry["api_key"] = api_key
        entry["model"] = spec.get("model") or (
            DEFAULT_LOCAL_MODELS[nation]
            if provider == "ollama"
            else PROVIDERS[provider]["default_model"]
        )
        config[nation] = entry
    return config


def _store_checkpoint(payload):
    global _latest_state
    with _state_lock:
        _latest_state = payload


def _resolve_report_llm(params):
    """Build the optional dedicated report model.

    Precedence: explicit Web UI choice > ``REPORT_PROVIDER``/.env > fall back
    to the winner's model. ``none`` disables the LLM report entirely. Returns
    ``(llm, label, enabled)``.
    """
    ui = (params.get("report_provider") or "").strip().lower()
    if ui == "none":
        return None, None, False

    env_conf = report_env_config()
    if env_conf and env_conf.get("provider") == "none":
        return None, None, False
    with _api_keys_lock:
        overrides = dict(_api_keys)

    if ui in ("", "auto", "winner"):
        conf = env_conf
    else:
        conf = {
            "provider": ui,
            "model": (params.get("report_model") or "").strip() or None,
            "api_key": None,
        }
    if not conf or conf.get("provider") not in PROVIDERS:
        return None, None, True

    provider = conf["provider"]
    model = conf.get("model")
    try:
        llm = build_chat_model(
            provider=provider,
            model=model,
            api_key=conf.get("api_key") or overrides.get(provider),
        )
    except Exception:
        # A bad report config must never break the game; fall back silently.
        return None, None, True
    label = f"{provider}/{model or PROVIDERS[provider]['default_model']}"
    return llm, label, True


def _inject(event):
    active = _active_events
    if active is not None:
        active.put(event)


def _autosave():
    with _state_lock:
        state = _latest_state
    if state is None:
        return None
    try:
        meta = write_save_payload(state, save_id=AUTOSAVE_ID, save_dir=SAVE_DIR)
        _inject({"type": "saved", "save": meta, "auto": True})
        return meta
    except Exception as exc:  # pragma: no cover - defensive
        _inject({"type": "error", "message": f"Autosave failed: {exc}"})
        return None


def _start_stream(max_turns, model_config, lang, resume=None, report_llm=None, report_label=None, report_enabled=True):
    """Acquire the run lock and stream a simulation (fresh or resumed)."""
    global _active_events, _latest_state

    if not _run_lock.acquire(blocking=False):
        return _error_stream("A simulation is already running.")

    _stop_event.clear()
    with _state_lock:
        _latest_state = None
    events: "queue.Queue" = queue.Queue()
    _active_events = events

    def worker():
        global _active_events
        try:
            run(
                max_turns=max_turns,
                verbose=False,
                on_event=lambda event: events.put(event),
                model_config=model_config,
                should_stop=_stop_event.is_set,
                lang=lang,
                resume=resume,
                on_checkpoint=_store_checkpoint,
                report_llm=report_llm,
                report_label=report_label,
                report_enabled=report_enabled,
            )
            if _stop_event.is_set():
                _autosave()
        except Exception as exc:
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            _active_events = None
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


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/report.js")
def report_js():
    return Response(
        (BASE_DIR / "report.js").read_text(encoding="utf-8"),
        media_type="application/javascript",
    )


@app.get("/api/providers")
def providers():
    with _api_keys_lock:
        overrides = dict(_api_keys)
    report_conf = report_env_config() or {}
    return {
        "providers": provider_status(overrides),
        "local_models": LOCAL_MODELS,
        "defaults": {
            "local": DEFAULT_LOCAL_MODELS,
            "api": DEFAULT_API_PROVIDERS,
            "env": default_model_config(),
            "report": {
                "provider": report_conf.get("provider"),
                "model": report_conf.get("model"),
            },
        },
    }


@app.post("/api/keys")
async def set_api_keys(request: Request):
    """Store API keys entered in the Web UI (in memory, local server only)."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)
    keys = payload.get("keys") if isinstance(payload, dict) else None
    if not isinstance(keys, dict):
        return JSONResponse({"error": "'keys' object is required."}, status_code=400)

    changes = {}
    for provider, value in keys.items():
        if provider not in PROVIDERS or provider == "ollama":
            continue
        # An empty value clears any previously stored key for that provider.
        changes[provider] = value.strip() if isinstance(value, str) else ""

    with _api_keys_lock:
        for provider, value in changes.items():
            if value:
                _api_keys[provider] = value
            else:
                _api_keys.pop(provider, None)
        overrides = dict(_api_keys)
    return {"providers": provider_status(overrides)}


@app.delete("/api/keys")
def clear_api_keys():
    with _api_keys_lock:
        _api_keys.clear()
    return {"providers": provider_status({})}


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

    lang = (params.get("lang") or "en").strip().lower()
    if lang not in {"en", "zh"}:
        lang = "en"

    report_llm, report_label, report_enabled = _resolve_report_llm(params)
    return _start_stream(
        max_turns,
        model_config,
        lang,
        resume=None,
        report_llm=report_llm,
        report_label=report_label,
        report_enabled=report_enabled,
    )


@app.get("/api/load")
async def load_simulation(request: Request):
    params = dict(request.query_params)
    save_id = (params.get("id") or "").strip()
    if not save_id:
        return _error_stream("A save id is required.")

    try:
        payload = load_game(save_id, save_dir=SAVE_DIR)
    except (FileNotFoundError, ValueError) as exc:
        return _error_stream(str(exc))

    raw_turns = params.get("max_turns")
    max_turns = None
    if raw_turns:
        try:
            max_turns = int(raw_turns)
        except ValueError:
            return _error_stream("max_turns must be an integer")
        max_turns = max(1, min(max_turns, 200))

    try:
        model_config = _resume_model_config(payload.get("models"))
    except ValueError as exc:
        return _error_stream(str(exc))

    lang = (params.get("lang") or payload.get("lang") or "en").strip().lower()
    if lang not in {"en", "zh"}:
        lang = "en"

    report_llm, report_label, report_enabled = _resolve_report_llm(params)
    return _start_stream(
        max_turns,
        model_config,
        lang,
        resume=payload,
        report_llm=report_llm,
        report_label=report_label,
        report_enabled=report_enabled,
    )


@app.get("/api/saves")
def list_simulation_saves():
    return {"saves": list_saves(save_dir=SAVE_DIR)}


@app.post("/api/save")
async def save_simulation(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    name = (body or {}).get("name") if isinstance(body, dict) else None

    with _state_lock:
        state = _latest_state
    if state is None:
        return JSONResponse(
            {"error": "No game state to save yet."}, status_code=400
        )
    payload = dict(state)
    if name:
        payload["name"] = name
    try:
        meta = write_save_payload(
            payload, save_id=new_save_id(name), save_dir=SAVE_DIR
        )
    except Exception as exc:
        return JSONResponse({"error": f"Save failed: {exc}"}, status_code=500)
    return {"save": meta}


@app.delete("/api/saves")
def delete_simulation_save(id: str):
    try:
        deleted = delete_save(id, save_dir=SAVE_DIR)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"deleted": deleted}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
