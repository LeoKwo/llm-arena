"""End-of-game news report generation.

Two flavours are produced:

* ``template_report`` — fully deterministic, localised (en/zh), built purely
  from the game timeline. Always available, never fails.
* ``llm_report`` — a "war correspondent" narrative written by one of the game
  models. Best-effort: any failure returns ``None`` so the caller falls back to
  the template.

``generate_report`` tries the LLM first, overlays its prose onto the template
structure (keeping the accurate timeline/outcome from the template), and marks
the result with ``source``.
"""

from __future__ import annotations

from typing import Optional

# Broadcast kinds that count as "major" events for the timeline / chart.
KEY_EVENT_KINDS = {
    "city_captured",
    "capital_captured",
    "capital_retaken",
    "capital_annexed",
    "faction_eliminated",
}


def _event_text(event: dict, lang: str) -> str:
    kind = event.get("kind")
    turn = event.get("turn", 0) + 1
    actor = event.get("actor")
    city = event.get("city")
    previous = event.get("previous")
    owner = event.get("owner")
    if lang == "zh":
        if kind == "city_captured":
            src = f"从 {previous} 手中" if previous else "从守军手中"
            return f"第 {turn} 回合：{actor} {src}攻占 {city}。"
        if kind == "capital_captured":
            return f"第 {turn} 回合：{actor} 攻陷 {owner} 的首都 {city}！"
        if kind == "capital_retaken":
            return f"第 {turn} 回合：{owner} 夺回首都 {city}。"
        if kind == "capital_annexed":
            return (
                f"第 {turn} 回合：{actor} 吞并 {owner}——首都 {city} "
                f"及 {owner} 全部城市易主。"
            )
        if kind == "faction_eliminated":
            return f"第 {turn} 回合：{actor} 被彻底消灭。"
    else:
        if kind == "city_captured":
            src = f" from {previous}" if previous else ""
            return f"Turn {turn}: {actor} captured {city}{src}."
        if kind == "capital_captured":
            return f"Turn {turn}: {actor} stormed {owner}'s capital {city}!"
        if kind == "capital_retaken":
            return f"Turn {turn}: {owner} recaptured the capital {city}."
        if kind == "capital_annexed":
            return (
                f"Turn {turn}: {actor} annexed {owner} — the capital {city} "
                f"and all of {owner}'s cities changed hands."
            )
        if kind == "faction_eliminated":
            return f"Turn {turn}: {actor} was eliminated."
    return f"Turn {turn}: {kind}."


def _rivalry(events: list[dict], world) -> Optional[tuple[str, str]]:
    tally: dict[tuple[str, str], int] = {}
    for event in events:
        kind = event.get("kind")
        if kind == "city_captured":
            pair = (event.get("actor"), event.get("previous"))
        elif kind in ("capital_captured", "capital_annexed"):
            pair = (event.get("actor"), event.get("owner"))
        else:
            continue
        a, b = pair
        if not a or not b or a == b:
            continue
        key = (a, b) if a < b else (b, a)
        tally[key] = tally.get(key, 0) + 1
    if not tally:
        return None
    # Most clashes first, then the coldest relationship as a tie-breaker.
    return min(
        tally,
        key=lambda pair: (-tally[pair], world.relation(pair[0], pair[1])),
    )


def build_digest(world, winner: Optional[str]) -> dict:
    timeline = world.timeline()
    events = [
        event for event in timeline["events"] if event.get("kind") in KEY_EVENT_KINDS
    ]
    score_history = timeline["score_history"]
    final_turn = score_history[-1]["turn"] if score_history else max(0, world.turn - 1)
    rankings = world.rankings()
    return {
        "winner": winner,
        "final_turn": final_turn,
        "max_turns": world.max_turns,
        "rankings": [(name, round(score, 1)) for name, score in rankings],
        "events": events,
        "rivalry": _rivalry(events, world),
        "score_history": score_history,
    }


# ------------------------------------------------------------------ template
def template_report(digest: dict, lang: str) -> dict:
    winner = digest.get("winner")
    final_turn = digest.get("final_turn", 0)
    rankings = digest.get("rankings", [])
    events = digest.get("events", [])
    rivalry = digest.get("rivalry")
    timeline = [
        {"turn": event.get("turn", 0), "text": _event_text(event, lang)}
        for event in events
    ]
    top = rankings[0] if rankings else (winner or "-", 0)
    runner = rankings[1] if len(rankings) > 1 else None
    rival_text = f"{rivalry[0]} / {rivalry[1]}" if rivalry else None

    if lang == "zh":
        headline = (
            f"{winner} 赢得欧洲争霸战" if winner else "欧洲争霸战：无人胜出"
        )
        dateline = f"LLM Arena · 共 {final_turn + 1} 回合"
        lead = (
            f"历时 {final_turn + 1} 个回合的欧洲六边形棋局落幕，"
            f"{winner or '没有任何势力'} 以 {int(top[1])} 分登顶。"
        )
        if rival_text:
            lead += f" 全场最激烈的对抗发生在 {rival_text} 之间。"
        paragraphs = []
        if runner is not None:
            paragraphs.append(
                f"最终排名方面，{top[0]} 以 {int(top[1])} 分领先，"
                f"{runner[0]} 以 {int(runner[1])} 分位居次席，"
                f"其余势力为 "
                + "、".join(
                    f"{name} {int(score)} 分" for name, score in rankings[2:]
                )
                + "。"
            )
        capitals = [
            event for event in events
            if event.get("kind") in ("capital_captured", "capital_annexed")
        ]
        eliminations = [
            event for event in events if event.get("kind") == "faction_eliminated"
        ]
        if capitals:
            paragraphs.append(
                "战局的关键转折在于首都攻防：" + " ".join(
                    _event_text(event, lang) for event in capitals
                )
            )
        if eliminations:
            paragraphs.append(
                "本局共有 "
                + "、".join(event.get("actor") for event in eliminations)
                + " 被彻底消灭。"
            )
        if not paragraphs:
            paragraphs.append("本局未爆发足以改写格局的重大战役。")
        outcome = (
            f"结果：{top[0]} 以 {int(top[1])} 分获胜。"
            if winner else "结果：达到回合上限，无单一胜者。"
        )
    else:
        headline = (
            f"{winner} wins the struggle for Europe"
            if winner
            else "The struggle for Europe ends without a victor"
        )
        dateline = f"LLM Arena · {final_turn + 1} turns"
        lead = (
            f"After {final_turn + 1} turns on the hexagonal map of Europe, "
            f"{winner or 'no single faction'} finished on top with "
            f"{int(top[1])} points."
        )
        if rival_text:
            lead += f" The fiercest fighting was between {rival_text}."
        paragraphs = []
        if runner is not None:
            paragraphs.append(
                f"On the final scoreboard {top[0]} led with {int(top[1])} points, "
                f"{runner[0]} followed on {int(runner[1])}, and the rest "
                + ", ".join(
                    f"{name} {int(score)}" for name, score in rankings[2:]
                )
                + "."
            )
        capitals = [
            event for event in events
            if event.get("kind") in ("capital_captured", "capital_annexed")
        ]
        eliminations = [
            event for event in events if event.get("kind") == "faction_eliminated"
        ]
        if capitals:
            paragraphs.append(
                "The turning points came at the capitals: " + " ".join(
                    _event_text(event, lang) for event in capitals
                )
            )
        if eliminations:
            paragraphs.append(
                "Eliminated this game: "
                + ", ".join(event.get("actor") for event in eliminations) + "."
            )
        if not paragraphs:
            paragraphs.append(
                "No decisive campaign reshaped the board this game."
            )
        outcome = (
            f"Result: {top[0]} wins with {int(top[1])} points."
            if winner
            else "Result: the turn limit was reached with no single winner."
        )

    return {
        "source": "template",
        "headline": headline,
        "dateline": dateline,
        "lead": lead,
        "paragraphs": paragraphs,
        "timeline": timeline,
        "outcome": outcome,
    }


# ----------------------------------------------------------------------- LLM
def _build_prompt(digest: dict, lang: str) -> str:
    winner = digest.get("winner")
    events = digest.get("events", [])
    lines = [
        "You are a war correspondent covering a strategy game between AI "
        "nations (Germany, France, United Kingdom, Soviet Union) on a hex map "
        "of Europe.",
        f"The game lasted {digest.get('final_turn', 0) + 1} turns. "
        f"The winner was {winner or 'nobody (turn limit reached)'}.",
        "Final standings: "
        + ", ".join(f"{name} {int(score)}" for name, score in digest.get("rankings", [])),
        "Major events:",
    ]
    for event in events:
        lines.append("- " + _event_text(event, "en"))
    if digest.get("rivalry"):
        lines.append(
            f"Main rivalry: {digest['rivalry'][0]} vs {digest['rivalry'][1]}."
        )
    if lang == "zh":
        lines.append(
            "\n请以战地记者的口吻，用简体中文写一篇短小的新闻通讯，"
            "包含双方、结果、经过与原因。严格按以下格式输出：\n"
            "Headline: <一行标题>\n"
            "Body:\n"
            "<3 到 5 个自然段，段与段之间用空行分隔>"
        )
    else:
        lines.append(
            "\nWrite a short news dispatch in English as a war correspondent, "
            "covering the belligerents, the outcome, and how and why it "
            "happened. Use exactly this format:\n"
            "Headline: <one line>\n"
            "Body:\n"
            "<3 to 5 paragraphs separated by blank lines>"
        )
    return "\n".join(lines)


def _extract_text(response) -> str:
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


def _parse_llm_output(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    headline = ""
    body = text
    lowered = text.lower()
    if "headline:" in lowered:
        index = lowered.index("headline:")
        after = text[index + len("headline:"):]
        first_nl = after.find("\n")
        if first_nl == -1:
            headline = after.strip()
            body = ""
        else:
            headline = after[:first_nl].strip()
            body = after[first_nl + 1:]
    else:
        lines = text.splitlines()
        headline = lines[0].strip()
        body = "\n".join(lines[1:])
    if "body:" in body.lower():
        index = body.lower().index("body:")
        body = body[index + len("body:"):]
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if not headline or not paragraphs:
        return None
    return {
        "headline": headline.strip().strip("*#").strip(),
        "lead": paragraphs[0],
        "paragraphs": paragraphs[1:],
    }


def llm_report(digest: dict, llm, lang: str) -> Optional[dict]:
    if llm is None:
        return None
    try:
        response = llm.invoke(_build_prompt(digest, lang))
    except Exception:
        return None
    return _parse_llm_output(_extract_text(response))


def generate_report(world, winner: Optional[str], llm=None, lang: str = "en") -> dict:
    """Build the end-of-game report, preferring the LLM narrative."""
    digest = build_digest(world, winner)
    report = template_report(digest, lang)
    if llm is not None:
        prose = llm_report(digest, llm, lang)
        if prose:
            report.update(prose)
            report["source"] = "llm"
    return report
