from types import SimpleNamespace

from engine.report import build_digest, generate_report, template_report


def _world_with_war(world):
    # Germany captures Paris, then France is wiped out.
    paris = world.cities["Paris"]
    paris.owner = "Germany"
    world._broadcast(
        "city_captured", actor="Germany", city="Paris", previous="France"
    )
    world._broadcast(
        "capital_captured", actor="Germany", owner="France", city="Paris"
    )
    world.cities["London"].owner = None
    world._check_elimination("France")
    world.advance()
    return world


def test_template_report_is_localised_and_complete(world):
    world = _world_with_war(world)
    report = template_report(build_digest(world, "Germany"), "en")
    assert report["source"] == "template"
    assert "Germany" in report["headline"]
    assert report["timeline"]
    assert any("Paris" in entry["text"] for entry in report["timeline"])
    assert "Result" in report["outcome"]

    zh = template_report(build_digest(world, "Germany"), "zh")
    assert "赢得" in zh["headline"]
    assert zh["timeline"][0]["text"].startswith("第")


def test_digest_detects_rivalry(world):
    world = _world_with_war(world)
    digest = build_digest(world, "Germany")
    assert digest["rivalry"] == tuple(sorted(("Germany", "France")))


def test_generate_report_falls_back_when_llm_fails(world):
    world = _world_with_war(world)

    class Boom:
        def invoke(self, prompt):
            raise RuntimeError("model offline")

    report = generate_report(world, "Germany", llm=Boom(), lang="en")
    assert report["source"] == "template"
    assert report["timeline"]


def test_generate_report_uses_llm_prose(world):
    world = _world_with_war(world)

    class FakeLLM:
        def invoke(self, prompt):
            assert "war correspondent" in prompt
            return SimpleNamespace(
                content=(
                    "Headline: Peace at last in Europe\n"
                    "Body:\nGermany rolled across the continent.\n\n"
                    "Paris fell within days, ending French resistance."
                )
            )

    report = generate_report(world, "Germany", llm=FakeLLM(), lang="en")
    assert report["source"] == "llm"
    assert report["headline"] == "Peace at last in Europe"
    assert report["lead"] == "Germany rolled across the continent."
    assert report["paragraphs"] == ["Paris fell within days, ending French resistance."]
    # The deterministic timeline is preserved regardless of LLM output.
    assert report["timeline"]
    assert report["outcome"]


def test_llm_output_without_format_is_rejected(world):
    world = _world_with_war(world)

    class Garbage:
        def invoke(self, prompt):
            return SimpleNamespace(content="   ")

    report = generate_report(world, "Germany", llm=Garbage(), lang="en")
    assert report["source"] == "template"


def test_template_report_handles_no_events(world):
    report = template_report(build_digest(world, None), "en")
    assert report["timeline"] == []
    assert report["outcome"]
