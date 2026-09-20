"""Advisor helpers for the human-controlled faction.

The faction's own LLM acts as a chief of staff: it proposes at least N options
(each with a rationale and a risk estimate), and later turns the commander's
chosen intent into a short, confirmed plan. Every LLM path has a deterministic
template fallback so the UI always has something to show (important for small
local models that do not reliably return structured output).
"""

from __future__ import annotations

import json
from typing import Optional

from agent.prompt import language_instruction

RISK_LEVELS = ("low", "medium", "high")

ATTACK_HINTS = ("attack", "capture", "assault", "siege", "invade", "strike", "raid",
                "进攻", "攻城", "入侵", "突袭", "夺取", "打")
DEFEND_HINTS = ("defend", "hold", "fortify", "garrison", "consolidate", "protect",
                "防守", "防御", "巩固", "守住", "保护")
ECON_HINTS = ("loot", "economy", "spawn", "build", "recruit", "supply", "resources",
              "掠夺", "经济", "生产", "招募", "补给", "资源")


def _text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _lang_note(lang: str) -> str:
    extra = language_instruction(lang)
    return ("\n\n" + extra) if extra else ""


def _first_json(text: str):
    """Extract the first JSON array or object from a model response."""
    if not text:
        return None
    text = text.strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            chunk = text[start:end + 1]
            try:
                return json.loads(chunk)
            except Exception:
                continue
    return None


# ------------------------------------------------------------------- risk
def assess_intent(world, name: str, text: str) -> dict:
    """Rough, purely heuristic risk estimate for an intent string."""
    lowered = (text or "").lower()
    own_units = world.units_of(name)
    strongest = max((u.resources for u in own_units), default=0)
    enemy_cities = [c for c in world.cities.values() if c.owner != name]
    nearest_defense = min(
        (world.city_defense(c) for c in enemy_cities), default=0
    )

    if any(h in lowered for h in ATTACK_HINTS):
        if strongest <= 0:
            return {"level": "high", "note": "You have no units to attack with."}
        if strongest < nearest_defense:
            return {
                "level": "high",
                "note": (
                    f"Your strongest unit ({int(strongest)}) is weaker than the "
                    f"weakest enemy city defence ({int(nearest_defense)}); the "
                    f"assault is likely to fail and lose the unit."
                ),
            }
        return {
            "level": "medium",
            "note": "An offensive can succeed but will cost resources.",
        }
    if any(h in lowered for h in DEFEND_HINTS):
        return {"level": "low", "note": "A defensive posture risks little."}
    if any(h in lowered for h in ECON_HINTS):
        return {"level": "low", "note": "Economic moves are low risk this turn."}
    return {"level": "medium", "note": "Outcome is uncertain; the AI will interpret it."}


# ------------------------------------------------------- template options
def _nearest_target_city(world, name):
    own = world.cities_of(name)
    if not own:
        return None
    origin = own[0].coords
    from world.hexmap import hex_distance

    targets = [c for c in world.cities.values() if c.owner != name]
    if not targets:
        return None
    return min(targets, key=lambda c: hex_distance(origin, c.coords))


def template_suggestions(world, name: str, count: int = 3, lang: str = "en") -> list:
    target = _nearest_target_city(world, name)
    own_cities = world.cities_of(name)
    capital = own_cities[0].city_name if own_cities else None
    own_units = world.units_of(name)
    options = []

    if target is not None:
        who = target.owner or "neutral"
        options.append(
            {
                "title": f"Assault {target.city_name}",
                "intent": (
                    f"Gather my forces and attack {target.city_name} ({who}), "
                    f"capturing it if possible."
                ),
                "rationale": (
                    f"{target.city_name} is the closest city not under my control "
                    f"(defence {int(world.city_defense(target))}); taking it adds "
                    f"50+ points and income."
                ),
                "risk": "high",
            }
        )
    options.append(
        {
            "title": "Consolidate and defend",
            "intent": (
                f"Keep my units together, reinforce {capital or 'my capital'} and "
                f"avoid risky attacks this turn."
            ),
            "rationale": "Preserves strength and protects the capital while the situation develops.",
            "risk": "low",
        }
    )
    options.append(
        {
            "title": "Build up and raid",
            "intent": (
                "Spend this turn on economy: create a new unit if affordable and "
                "loot nearby tiles with existing units."
            ),
            "rationale": "More resources and units compound into a stronger position later.",
            "risk": "low",
        }
    )
    if own_units:
        options.append(
            {
                "title": "Forced-march raid",
                "intent": (
                    "Use forced march to reach a valuable target this turn, "
                    "accepting the resource cost."
                ),
                "rationale": "Extra movement can catch a defender off guard, at the cost of resources.",
                "risk": "medium",
            }
        )
    out = options[:count]
    # Guarantee at least ``count`` options even in a degenerate world state.
    index = 0
    while len(out) < count:
        out.append(
            {
                "title": f"Hold position #{index + 1}",
                "intent": "End this turn conservatively without overextending.",
                "rationale": "A safe default when nothing else is available.",
                "risk": "low",
            }
        )
        index += 1
    return out


# ----------------------------------------------------------- LLM options
def _options_prompt(world, name, observation, count, lang):
    return (
        f"You are the chief of staff advising {name}, a faction in a turn-based "
        f"conquest game.\n\nCurrent situation:\n{observation}\n\n"
        f"Propose {count} distinct strategic directions for this turn. Each option "
        "must be a concrete, actionable order the faction can attempt now.\n"
        "Return ONLY a JSON array, each item with keys: "
        '"title", "intent", "rationale", "risk" (one of low/medium/high).'
        + _lang_note(lang)
    )


def _normalise_option(raw):
    if not isinstance(raw, dict):
        return None
    intent = str(raw.get("intent") or "").strip()
    if not intent:
        return None
    title = str(raw.get("title") or intent[:48]).strip()
    rationale = str(raw.get("rationale") or "").strip()
    risk = str(raw.get("risk") or "").strip().lower()
    if risk not in RISK_LEVELS:
        risk = ""
    return {"title": title, "intent": intent, "rationale": rationale, "risk": risk}


def build_suggestions(world, name, observation, llm, lang="en", count=3):
    """Return at least ``count`` suggestions, LLM-first with a template fallback."""
    fallback = template_suggestions(world, name, count, lang)
    if llm is None:
        return fallback

    try:
        response = llm.invoke(_options_prompt(world, name, observation, count, lang))
    except Exception:
        return fallback

    parsed = _first_json(_text(response))
    if isinstance(parsed, dict):
        parsed = parsed.get("options") or parsed.get("suggestions") or []
    options = []
    if isinstance(parsed, list):
        for raw in parsed:
            option = _normalise_option(raw)
            if option is not None:
                options.append(option)
    if len(options) < count:
        titles = {o["title"] for o in options}
        for extra in fallback:
            if len(options) >= count:
                break
            if extra["title"] not in titles:
                options.append(extra)

    for option in options:
        if not option.get("risk"):
            option["risk"] = assess_intent(world, name, option["intent"])["level"]
        if not option.get("rationale"):
            option["rationale"] = assess_intent(world, name, option["intent"])["note"]
    return options[: max(count, len(options))]


# ------------------------------------------------------------- plan summary
def _plan_prompt(world, name, intent, observation, lang):
    return (
        f"You are the chief of staff of {name}. The commander has decided:\n"
        f'"{intent}"\n\nCurrent situation:\n{observation}\n\n'
        "Summarise this into a short operational plan. Return ONLY a JSON object "
        'with keys: "objective" (one sentence), "steps" (array of short strings), '
        '"risks" (array of short strings). If the order is likely to fail or cost '
        "the faction dearly, say so plainly in the risks."
        + _lang_note(lang)
    )


def template_plan(world, name, intent, assessment):
    return {
        "objective": intent,
        "steps": [
            "Follow the commander's directive to the best of the faction's ability.",
            "Use available units and action points; end the turn when done.",
        ],
        "risks": [assessment.get("note", "")] if assessment.get("note") else [],
    }


def summarize_plan(world, name, intent, observation, llm, lang="en"):
    assessment = assess_intent(world, name, intent)
    plan = template_plan(world, name, intent, assessment)
    if llm is not None:
        try:
            response = llm.invoke(_plan_prompt(world, name, intent, observation, lang))
            parsed = _first_json(_text(response))
            if isinstance(parsed, dict) and parsed.get("objective"):
                plan = {
                    "objective": str(parsed.get("objective")).strip(),
                    "steps": [str(s).strip() for s in (parsed.get("steps") or []) if str(s).strip()],
                    "risks": [str(s).strip() for s in (parsed.get("risks") or []) if str(s).strip()],
                }
        except Exception:
            pass

    warning = ""
    if assessment.get("level") == "high":
        warning = assessment.get("note", "")
    return {
        "objective": plan["objective"],
        "steps": plan["steps"],
        "risks": plan["risks"],
        "risk": assessment.get("level", "medium"),
        "warning": warning,
    }
