import threading
import time

from engine.human import HumanController

FACTIONS = {"Germany", "France", "United Kingdom", "Soviet Union"}


def _wait_for(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_join_validates_and_is_permanent():
    controller = HumanController()
    assert controller.join("Atlantis", FACTIONS)["ok"] is False
    assert controller.join("Germany", FACTIONS)["ok"] is True
    assert controller.is_human("Germany") is True
    # Committed for the rest of the game: no switching.
    assert controller.join("France", FACTIONS)["ok"] is False
    assert controller.is_human("France") is False


def test_restore_rejoins_saved_faction():
    controller = HumanController()
    controller.restore("France", joined_turn=4)
    assert controller.is_human("France")
    assert controller.snapshot()["joined"] is True


def test_intent_roundtrip():
    controller = HumanController(timeout=5)
    result = {}

    def worker():
        result["intent"] = controller.request_intent({"turn": 1})

    thread = threading.Thread(target=worker)
    thread.start()
    assert _wait_for(lambda: controller.snapshot()["phase"] == "awaiting_intent")
    assert controller.submit_intent({"choice": 1}) is True
    thread.join(2)
    assert result["intent"] == {"choice": 1}
    assert controller.snapshot()["phase"] == "idle"


def test_intent_timeout():
    controller = HumanController(timeout=0.1)
    assert controller.request_intent({"turn": 0}) == {"kind": "timeout"}


def test_confirm_roundtrip():
    controller = HumanController(timeout=5)
    result = {}

    def worker():
        result["confirm"] = controller.request_confirm({"turn": 1, "plan": {}})

    thread = threading.Thread(target=worker)
    thread.start()
    assert _wait_for(lambda: controller.snapshot()["phase"] == "awaiting_confirm")
    assert controller.submit_confirm({"accept": True}) is True
    thread.join(2)
    assert result["confirm"] == {"accept": True}


def test_autopilot_unblocks():
    controller = HumanController(timeout=5)
    result = {}

    def worker():
        result["intent"] = controller.request_intent({"turn": 0})

    thread = threading.Thread(target=worker)
    thread.start()
    assert _wait_for(lambda: controller.snapshot()["phase"] == "awaiting_intent")
    assert controller.autopilot() is True
    thread.join(2)
    assert result["intent"] == {"kind": "autopilot"}


def test_stop_unblocks_immediately():
    controller = HumanController(timeout=5, should_stop=lambda: True)
    assert controller.request_intent({"turn": 0}) == {"kind": "stop"}


def test_submit_rejected_out_of_phase():
    controller = HumanController()
    assert controller.submit_intent({"choice": 0}) is False
    assert controller.submit_confirm({"accept": True}) is False
    assert controller.autopilot() is False


def test_snapshot_exposes_pending_prompt():
    controller = HumanController(timeout=5)
    thread = threading.Thread(target=lambda: controller.request_intent({"turn": 2, "suggestions": []}))
    thread.start()
    assert _wait_for(lambda: controller.snapshot()["phase"] == "awaiting_intent")
    snap = controller.snapshot()
    assert snap["prompt"]["turn"] == 2
    controller.autopilot()
    thread.join(2)
