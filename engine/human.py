"""Human-in-the-loop controller for LLM Arena.

A single human can take over one faction for the rest of a game. While it is
that faction's turn, the simulation thread pauses here and waits for the human
to pick a suggestion (or write their own order) and confirm the plan; if they
do not respond in time, the turn falls back to the LLM ("autopilot").

The controller is deliberately transport-agnostic: it emits events through a
callback and is driven by ``submit_intent`` / ``submit_confirm`` from the web
layer. All shared state is guarded by a lock because it is touched from both
the simulation worker thread and the FastAPI thread pool.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

POLL_INTERVAL = 0.05


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, default))


DEFAULT_TIMEOUT = _env_float("HUMAN_TIMEOUT", 90.0)
DEFAULT_SUGGESTIONS = _env_int("SUGGESTION_COUNT", 3)


class HumanController:
    def __init__(
        self,
        emit: Optional[Callable[[dict], None]] = None,
        timeout: Optional[float] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        suggest_count: Optional[int] = None,
    ) -> None:
        self.emit = emit or (lambda event: None)
        self.timeout = DEFAULT_TIMEOUT if timeout is None else float(timeout)
        self.should_stop = should_stop or (lambda: False)
        self.suggest_count = suggest_count or DEFAULT_SUGGESTIONS

        self.faction: Optional[str] = None
        self.joined_turn: Optional[int] = None
        self.phase = "idle"  # idle | awaiting_intent | awaiting_confirm
        self.turn: Optional[int] = None
        self.prompt: Optional[dict] = None
        self.plan: Optional[dict] = None

        self._lock = threading.Lock()
        self._event = threading.Event()
        self._intent: Optional[dict] = None
        self._confirm: Optional[dict] = None
        self._autopilot = False

    # ------------------------------------------------------------- state
    def is_human(self, name: str) -> bool:
        return self.faction == name

    def join(self, faction: str, valid_factions, turn: int = 0) -> dict:
        with self._lock:
            if self.faction is not None:
                return {"ok": False, "error": "Already participating in this game."}
            if faction not in set(valid_factions):
                return {"ok": False, "error": f"Unknown faction '{faction}'."}
            self.faction = faction
            self.joined_turn = turn
        self.emit({"type": "human_joined", "agent": faction, "turn": turn})
        return {"ok": True, "faction": faction, "turn": turn}

    def restore(self, faction: Optional[str], joined_turn: int = 0) -> None:
        """Re-attach a faction when resuming a saved game (no validation)."""
        if not faction:
            return
        with self._lock:
            self.faction = faction
            self.joined_turn = joined_turn
        self.emit({"type": "human_joined", "agent": faction, "turn": joined_turn})

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "joined": self.faction is not None,
                "faction": self.faction,
                "joined_turn": self.joined_turn,
                "phase": self.phase,
                "turn": self.turn,
                "prompt": self.prompt,
                "plan": self.plan,
                "timeout": self.timeout,
            }

    # ------------------------------------------------------- wait helpers
    def _reset_signals(self) -> None:
        self._intent = None
        self._confirm = None
        self._autopilot = False
        self._event.clear()

    def _deadline(self, timeout: Optional[float]):
        value = self.timeout if timeout is None else float(timeout)
        if value is None or value <= 0:
            return None  # wait forever
        return time.monotonic() + value

    def _block(self, deadline) -> str:
        """Wait for a signal, stop request, or timeout. Returns the reason."""
        while True:
            if self.should_stop():
                return "stop"
            if self._event.wait(POLL_INTERVAL):
                return "signal"
            if deadline is not None and time.monotonic() >= deadline:
                return "timeout"

    def _outcome(self, reason: str, slot: Optional[dict], kind_when_missing: str) -> dict:
        if self._autopilot:
            return {"kind": "autopilot"}
        if slot is not None:
            return slot
        if reason == "stop":
            return {"kind": "stop"}
        return {"kind": kind_when_missing}

    # --------------------------------------------------------- intent
    def request_intent(self, prompt: dict, timeout: Optional[float] = None) -> dict:
        deadline = self._deadline(timeout)
        with self._lock:
            self._reset_signals()
            self.phase = "awaiting_intent"
            self.turn = prompt.get("turn")
            self.prompt = prompt
        self.emit({"type": "human_prompt", **prompt})
        reason = self._block(deadline)
        with self._lock:
            slot = self._intent
            self.phase = "idle"
            self.prompt = None
            out = self._outcome(reason, slot, "timeout")
        if out.get("kind") == "timeout":
            self.emit({"type": "human_timeout", "agent": self.faction, "turn": prompt.get("turn")})
        return out

    def submit_intent(self, payload: dict) -> bool:
        with self._lock:
            if self.phase != "awaiting_intent":
                return False
            self._intent = dict(payload or {})
            self._event.set()
        return True

    # --------------------------------------------------------- confirm
    def request_confirm(self, plan: dict, timeout: Optional[float] = None) -> dict:
        deadline = self._deadline(timeout)
        with self._lock:
            self._reset_signals()
            self.phase = "awaiting_confirm"
            self.turn = plan.get("turn")
            self.plan = plan
        self.emit({"type": "human_plan", **plan})
        reason = self._block(deadline)
        with self._lock:
            slot = self._confirm
            self.phase = "idle"
            self.plan = None
            out = self._outcome(reason, slot, "timeout")
        if out.get("kind") == "timeout":
            self.emit({"type": "human_timeout", "agent": self.faction, "turn": plan.get("turn")})
        return out

    def submit_confirm(self, payload: dict) -> bool:
        with self._lock:
            if self.phase != "awaiting_confirm":
                return False
            self._confirm = dict(payload or {})
            self._event.set()
        return True

    def autopilot(self) -> bool:
        with self._lock:
            if self.phase == "idle":
                return False
            self._autopilot = True
            self._event.set()
        return True

    def joined_event(self) -> dict:
        return {"type": "human_joined", "agent": self.faction, "turn": self.joined_turn}
