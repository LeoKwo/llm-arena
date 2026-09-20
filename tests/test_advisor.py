from types import SimpleNamespace

from engine.advisor import (
    assess_intent,
    build_suggestions,
    summarize_plan,
    template_suggestions,
)


def test_template_suggestions_always_meet_minimum(world):
    options = template_suggestions(world, "Germany", 4)
    assert len(options) >= 4
    assert all(option["intent"] for option in options)
    assert all(option["title"] for option in options)


def test_build_suggestions_falls_back_without_llm(world):
    options = build_suggestions(world, "Germany", "obs", None, "en", 3)
    assert len(options) >= 3
    assert all(option["risk"] in ("low", "medium", "high") for option in options)


def test_build_suggestions_parses_llm_json(world):
    class LLM:
        def invoke(self, prompt):
            return SimpleNamespace(
                content=(
                    '[{"title":"A","intent":"Attack X","rationale":"because",'
                    '"risk":"high"},'
                    '{"title":"B","intent":"Defend Y","rationale":"safe",'
                    '"risk":"low"},'
                    '{"title":"C","intent":"Grow economy","rationale":"grow",'
                    '"risk":"low"}]'
                )
            )

    options = build_suggestions(world, "Germany", "obs", LLM(), "en", 3)
    assert len(options) == 3
    assert options[0]["title"] == "A"
    assert options[0]["risk"] == "high"


def test_build_suggestions_tops_up_bad_json(world):
    class LLM:
        def invoke(self, prompt):
            return SimpleNamespace(content="sorry, I cannot do that")

    options = build_suggestions(world, "Germany", "obs", LLM(), "en", 3)
    assert len(options) >= 3


def test_build_suggestions_survives_llm_error(world):
    class LLM:
        def invoke(self, prompt):
            raise RuntimeError("model offline")

    options = build_suggestions(world, "Germany", "obs", LLM(), "en", 3)
    assert len(options) >= 3


def test_assess_intent_levels(world):
    # Germany starts with no units, so any attack is high risk.
    assert assess_intent(world, "Germany", "attack Paris now")["level"] == "high"
    assert assess_intent(world, "Germany", "hold and defend the capital")["level"] == "low"
    assert assess_intent(world, "Germany", "loot tiles and grow the economy")["level"] == "low"


def test_summarize_plan_template_warns_on_high_risk(world):
    plan = summarize_plan(world, "Germany", "attack Paris now", "obs", None, "en")
    assert plan["objective"]
    assert plan["risk"] == "high"
    assert plan["warning"]


def test_summarize_plan_uses_llm_json(world):
    class LLM:
        def invoke(self, prompt):
            return SimpleNamespace(
                content='{"objective":"Take Paris","steps":["march east"],"risks":["losses"]}'
            )

    plan = summarize_plan(world, "Germany", "attack Paris", "obs", LLM(), "en")
    assert plan["objective"] == "Take Paris"
    assert plan["steps"] == ["march east"]
    assert plan["risks"] == ["losses"]


def test_summarize_plan_survives_llm_error(world):
    class LLM:
        def invoke(self, prompt):
            raise RuntimeError("boom")

    plan = summarize_plan(world, "Germany", "defend", "obs", LLM(), "en")
    assert plan["objective"] == "defend"
    assert plan["risk"] == "low"
