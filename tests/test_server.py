import pytest
from fastapi.testclient import TestClient

import web.server as server
from engine import savegame
from tests.fakes import FakeMemory
from world.environment import build_default_world

OLLAMA_PARAMS = "germany=ollama&france=ollama&uk=ollama&ussr=ollama"


def _make_fake_run(captured):
    def fake_run(**kwargs):
        captured.append(kwargs)
        world = build_default_world(seed="server-seed")
        world.max_turns = kwargs.get("max_turns") or 5
        on_event = kwargs.get("on_event")
        on_checkpoint = kwargs.get("on_checkpoint")
        if on_event:
            on_event(
                {
                    "type": "init",
                    "nations": {},
                    "world": world.snapshot(),
                    "max_turns": world.max_turns,
                    "resumed": bool(kwargs.get("resume")),
                }
            )
        if on_checkpoint:
            agents = {
                name: {"provider": "ollama", "model": "fake", "memory": FakeMemory()}
                for name in world.agents
            }
            on_checkpoint(
                savegame.build_save_payload(world, agents, {"lang": "en"})
            )
        if on_event:
            on_event(
                {
                    "type": "end",
                    "winner": "Germany",
                    "world": world.snapshot(),
                    "rankings": world.rankings(),
                    "timeline": world.timeline(),
                    "report": {"source": "template", "timeline": []},
                }
            )
        return "Germany"

    return fake_run


@pytest.fixture
def client(tmp_path, monkeypatch):
    save_dir = tmp_path / "saves"
    save_dir.mkdir()
    monkeypatch.setattr(server, "SAVE_DIR", save_dir)
    captured = []
    monkeypatch.setattr(server, "run", _make_fake_run(captured))
    return TestClient(server.app), captured


def test_index_and_report_js_are_served(client):
    c, _ = client
    assert c.get("/").status_code == 200
    report = c.get("/report.js")
    assert report.status_code == 200
    assert "buildScoreChartSvg" in report.text


def test_run_streams_lifecycle_and_save_roundtrip(client):
    c, _ = client
    resp = c.get(f"/api/run?max_turns=2&{OLLAMA_PARAMS}")
    assert resp.status_code == 200
    assert '"type": "init"' in resp.text
    assert '"type": "end"' in resp.text

    saved = c.post("/api/save", json={"name": "unit test"})
    assert saved.status_code == 200
    save_id = saved.json()["save"]["id"]

    listed = c.get("/api/saves").json()["saves"]
    assert [s["id"] for s in listed] == [save_id]

    deleted = c.request("DELETE", f"/api/saves?id={save_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert c.get("/api/saves").json()["saves"] == []


def test_save_without_state_is_rejected(client):
    c, _ = client
    server._latest_state = None
    resp = c.post("/api/save", json={})
    assert resp.status_code == 400


def test_load_resumes_a_saved_game(client):
    c, captured = client
    c.get(f"/api/run?max_turns=2&{OLLAMA_PARAMS}")
    save_id = c.post("/api/save", json={"name": "resume me"}).json()["save"]["id"]

    captured.clear()
    resp = c.get(f"/api/load?id={save_id}&max_turns=4")
    assert resp.status_code == 200
    assert captured
    resume = captured[-1]["resume"]
    assert resume is not None
    assert resume["id"] == save_id
    assert captured[-1]["max_turns"] == 4


def test_providers_expose_report_defaults(client, monkeypatch):
    monkeypatch.delenv("REPORT_PROVIDER", raising=False)
    c, _ = client
    data = c.get("/api/providers").json()
    assert "report" in data["defaults"]


def test_resolve_report_llm_precedence(monkeypatch):
    monkeypatch.delenv("REPORT_PROVIDER", raising=False)
    monkeypatch.delenv("REPORT_MODEL", raising=False)
    assert server._resolve_report_llm({"report_provider": "none"}) == (None, None, False)
    assert server._resolve_report_llm({}) == (None, None, True)

    monkeypatch.setenv("REPORT_PROVIDER", "ollama")
    monkeypatch.setenv("REPORT_MODEL", "qwen3.5:latest")
    llm, label, enabled = server._resolve_report_llm({})
    assert llm is not None
    assert label == "ollama/qwen3.5:latest"
    assert enabled is True

    # An explicit UI choice overrides the .env value.
    llm, label, enabled = server._resolve_report_llm({"report_provider": "ollama"})
    assert llm is not None
    assert label == "ollama/qwen3.5:latest"
    assert enabled is True

    # REPORT_PROVIDER=none in .env disables the AI report.
    monkeypatch.setenv("REPORT_PROVIDER", "none")
    assert server._resolve_report_llm({}) == (None, None, False)


def test_load_unknown_save_errors(client):
    c, _ = client
    resp = c.get("/api/load?id=missing-save")
    assert resp.status_code == 200  # SSE stream carries the error event
    assert "not found" in resp.text.lower()
