from engine import savegame
from engine.simulation_loop import run
from tests.fakes import end_turn_action, make_fake_agent_factory


def _actions(name, turn):
    return [end_turn_action(name)]


def test_resume_continues_from_saved_turn(save_dir):
    checkpoints = []
    run(
        max_turns=2,
        verbose=False,
        embeddings=object(),
        agent_factory=make_fake_agent_factory(_actions),
        on_checkpoint=checkpoints.append,
    )
    # Checkpoint after turn 1: the next turn to play is 2.
    payload = checkpoints[-1]
    assert payload["world"]["turn"] == 2
    assert [p["turn"] for p in payload["world"]["score_history"]] == [0, 1]

    savegame.write_save_payload(payload, save_id="mid", save_dir=save_dir)
    loaded = savegame.load_game("mid", save_dir=save_dir)

    created = {}
    base_factory = make_fake_agent_factory(_actions)

    def factory(*args, **kwargs):
        agents = base_factory(*args, **kwargs)
        created.update(agents)
        return agents

    events = []
    run(
        max_turns=4,
        verbose=False,
        on_event=events.append,
        embeddings=object(),
        agent_factory=factory,
        resume=loaded,
        lang="en",
    )

    assert events[0]["type"] == "init"
    assert events[0]["resumed"] is True
    assert [e["turn"] for e in events if e["type"] == "turn_start"] == [2, 3]

    end = next(e for e in events if e["type"] == "end")
    assert [p["turn"] for p in end["timeline"]["score_history"]] == [0, 1, 2, 3]
    assert end["world"]["turn"] == 4
    # Saved memory was carried into the freshly-built agents.
    assert created["Germany"]["memory"].memory_items


def test_resume_max_turns_cannot_truncate_below_saved_turn(save_dir):
    checkpoints = []
    run(
        max_turns=3,
        verbose=False,
        embeddings=object(),
        agent_factory=make_fake_agent_factory(_actions),
        on_checkpoint=checkpoints.append,
    )
    payload = checkpoints[-1]  # next_turn == 3
    savegame.write_save_payload(payload, save_id="late", save_dir=save_dir)
    loaded = savegame.load_game("late", save_dir=save_dir)

    events = []
    run(
        max_turns=1,  # smaller than the save's turn: must be clamped up
        verbose=False,
        on_event=events.append,
        embeddings=object(),
        agent_factory=make_fake_agent_factory(_actions),
        resume=loaded,
    )
    # No turn ran, and the game ended without corrupting the timeline.
    assert not [e for e in events if e["type"] == "turn_start"]
    end = next(e for e in events if e["type"] == "end")
    assert [p["turn"] for p in end["timeline"]["score_history"]] == [0, 1, 2]
