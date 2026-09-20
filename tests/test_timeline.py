def test_broadcast_writes_turn_to_both_logs(world):
    world.turn = 5
    world._broadcast("city_captured", actor="Germany", city="Paris", previous="France")
    assert world.event_log[-1]["kind"] == "city_captured"
    assert world.event_log[-1]["turn"] == 5
    assert world.broadcasts[-1]["turn"] == 5

    drained = world.take_broadcasts()
    assert len(drained) == 1
    # Draining the live-feed queue must not lose the persistent log.
    assert len(world.event_log) == 1
    assert world.broadcasts == []


def test_advance_records_one_score_point_per_turn(world):
    world.turn = 0
    world.advance()
    assert world.turn == 1
    assert len(world.score_history) == 1
    assert world.score_history[0]["turn"] == 0
    assert set(world.score_history[0]["scores"]) == set(world.agents)

    world.advance()
    assert world.turn == 2
    assert [p["turn"] for p in world.score_history] == [0, 1]


def test_timeline_returns_serialisable_copies(world):
    world.turn = 1
    world._broadcast("faction_eliminated", actor="France")
    world.advance()

    timeline = world.timeline()
    assert set(timeline) == {"score_history", "events", "history"}
    assert timeline["events"]
    # Mutating the returned copy must not affect the world.
    timeline["events"][0]["kind"] = "mutated"
    assert world.event_log[0]["kind"] == "faction_eliminated"
