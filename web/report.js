/* Pure front-end helpers for the end-of-game report.
 *
 * Kept dependency-free and DOM-free so it can run both in the browser (loaded
 * via <script src="/report.js">) and under `node --test`.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.ArenaReport = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const PALETTE = ["#e06c75", "#61afef", "#98c379", "#e5c07b", "#c678dd", "#56b6c2"];
  const MAJOR_KINDS = [
    "city_captured",
    "capital_captured",
    "capital_fallen",
    "capital_retaken",
    "capital_annexed",
    "faction_eliminated",
  ];

  const EN = {
    bc_city_captured: (a, c, f) => `${a} captured ${c}${f ? ` from ${f}` : ""}!`,
    bc_capital_captured: (o, c, by) => `${o}'s capital ${c} has fallen to ${by}!`,
    bc_capital_retaken: (o, c) => `${o} recaptured its capital ${c}!`,
    bc_capital_annexed: (a, o, c) =>
      `${a} annexed ${o} (capital ${c} and all its cities)!`,
    bc_eliminated: (a) => `${a} has been eliminated!`,
    bc_unit_destroyed: (a, u, v) => `${a} destroyed ${v}'s unit ${u}!`,
    report_source_llm: "AI dispatch",
    report_source_template: "Auto-generated",
    report_timeline: "Timeline of major events",
  };

  function defaultT(key, ...args) {
    const entry = EN[key];
    return typeof entry === "function" ? entry(...args) : entry !== undefined ? entry : key;
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatBroadcast(event, t) {
    const d = event || {};
    const tr = typeof t === "function" ? t : defaultT;
    switch (d.kind) {
      case "city_captured":
        return tr("bc_city_captured", d.actor, d.city, d.previous || "");
      case "capital_captured":
      case "capital_fallen":
        return tr("bc_capital_captured", d.owner, d.city, d.actor || d.by);
      case "capital_retaken":
        return tr("bc_capital_retaken", d.owner, d.city);
      case "capital_annexed":
        return tr("bc_capital_annexed", d.actor, d.owner, d.city);
      case "faction_eliminated":
        return tr("bc_eliminated", d.actor);
      case "unit_destroyed":
        return tr("bc_unit_destroyed", d.actor, d.unit, d.victim);
      default:
        return d.kind || "";
    }
  }

  function niceCeil(value) {
    if (!(value > 0)) return 10;
    const exp = Math.floor(Math.log10(value));
    const base = Math.pow(10, exp);
    const frac = value / base;
    const step = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
    return step * base;
  }

  function makeColorFor() {
    const assigned = {};
    return function colorFor(name) {
      if (!assigned[name]) {
        assigned[name] = PALETTE[Object.keys(assigned).length % PALETTE.length];
      }
      return assigned[name];
    };
  }

  function scoreAt(point, name) {
    return point && point.scores ? Number(point.scores[name] || 0) : 0;
  }

  function buildScoreChartSvg(timeline, opts) {
    opts = opts || {};
    const history = (timeline && timeline.score_history) || [];
    const events = (timeline && timeline.events) || [];
    const width = opts.width || 760;
    const height = opts.height || 320;
    const margin = opts.margin || { left: 54, right: 18, top: 36, bottom: 36 };
    const innerW = Math.max(1, width - margin.left - margin.right);
    const innerH = Math.max(1, height - margin.top - margin.bottom);
    const tr = typeof opts.t === "function" ? opts.t : defaultT;
    if (!history.length) {
      return `<div class="chart-empty">${escapeHtml(
        opts.emptyText || "No score history recorded."
      )}</div>`;
    }

    const names = [];
    const seen = {};
    (opts.nations || []).forEach((n) => {
      if (!seen[n]) { seen[n] = 1; names.push(n); }
    });
    history.forEach((point) =>
      Object.keys(point.scores || {}).forEach((n) => {
        if (!seen[n]) { seen[n] = 1; names.push(n); }
      })
    );

    const turns = history.map((p) => p.turn || 0);
    const minTurn = Math.min.apply(null, turns);
    const maxTurn = Math.max.apply(null, turns);
    const span = Math.max(1, maxTurn - minTurn);
    let maxScore = 0;
    history.forEach((p) =>
      names.forEach((n) => { maxScore = Math.max(maxScore, scoreAt(p, n)); })
    );
    const yMax = niceCeil(Math.max(10, maxScore));
    const x = (turn) => margin.left + ((turn - minTurn) / span) * innerW;
    const y = (val) => margin.top + (1 - val / yMax) * innerH;
    const colorFor = opts.colorFor || makeColorFor();

    const parts = [];
    parts.push(
      `<svg class="scorechart" viewBox="0 0 ${width} ${height}" ` +
        `preserveAspectRatio="xMidYMid meet" role="img">`
    );

    // Horizontal grid + y labels.
    for (let i = 0; i <= 4; i++) {
      const value = (yMax / 4) * i;
      const gy = y(value);
      parts.push(
        `<line x1="${margin.left}" y1="${gy.toFixed(1)}" x2="${(
          margin.left + innerW
        ).toFixed(1)}" y2="${gy.toFixed(1)}" class="chart-grid"/>`
      );
      parts.push(
        `<text x="${margin.left - 8}" y="${(gy + 3).toFixed(1)}" ` +
          `class="chart-axis" text-anchor="end">${Math.round(value)}</text>`
      );
    }

    // X ticks.
    const tickStep = Math.max(1, Math.ceil((span + 1) / 10));
    for (let turn = minTurn; turn <= maxTurn; turn += tickStep) {
      const gx = x(turn);
      parts.push(
        `<line x1="${gx.toFixed(1)}" y1="${margin.top}" x2="${gx.toFixed(
          1
        )}" y2="${(margin.top + innerH).toFixed(1)}" class="chart-grid faint"/>`
      );
      parts.push(
        `<text x="${gx.toFixed(1)}" y="${(
          margin.top + innerH + 20
        ).toFixed(1)}" class="chart-axis" text-anchor="middle">T${turn + 1}</text>`
      );
    }

    // One polyline per faction.
    names.forEach((name) => {
      const color = colorFor(name);
      const points = history
        .map((p) => `${x(p.turn || 0).toFixed(1)},${y(scoreAt(p, name)).toFixed(1)}`)
        .join(" ");
      parts.push(
        `<polyline class="chart-line" points="${points}" fill="none" ` +
          `stroke="${color}" stroke-width="2.2"/>`
      );
    });

    // Major-event markers. Group events by turn so clustered events share a line.
    const byTurn = {};
    events
      .filter((e) => MAJOR_KINDS.indexOf(e.kind) !== -1)
      .forEach((e) => {
        const key = Math.min(Math.max(e.turn || 0, minTurn), maxTurn);
        (byTurn[key] = byTurn[key] || []).push(e);
      });
    Object.keys(byTurn)
      .map(Number)
      .sort((a, b) => a - b)
      .forEach((turn) => {
        const gx = x(turn);
        const group = byTurn[turn];
        const label = group
          .map((e) => `${formatBroadcast(e, tr)} (T${(e.turn || 0) + 1})`)
          .join("\n");
        const point = history.reduce(
          (best, p) => (Math.abs((p.turn || 0) - turn) < Math.abs(best.turn - turn) ? p : best),
          history[0]
        );
        parts.push(`<g class="chart-marker"><title>${escapeHtml(label)}</title>`);
        parts.push(
          `<line x1="${gx.toFixed(1)}" y1="${margin.top}" x2="${gx.toFixed(
            1
          )}" y2="${(margin.top + innerH).toFixed(1)}" class="chart-event"/>`
        );
        group.forEach((e) => {
          const actor = e.actor || e.owner;
          if (!actor || names.indexOf(actor) === -1) return;
          parts.push(
            `<circle cx="${gx.toFixed(1)}" cy="${y(scoreAt(point, actor)).toFixed(
              1
            )}" r="3.6" fill="${colorFor(actor)}" stroke="#0f1115" stroke-width="1.2"/>`
          );
        });
        parts.push("</g>");
      });

    // Legend.
    let lx = margin.left;
    names.forEach((name) => {
      const color = colorFor(name);
      parts.push(
        `<rect x="${lx}" y="10" width="10" height="10" rx="2" fill="${color}"/>` +
          `<text x="${lx + 14}" y="19" class="chart-axis">${escapeHtml(name)}</text>`
      );
      lx += 14 + String(name).length * 7 + 18;
    });

    parts.push("</svg>");
    return parts.join("");
  }

  function buildReportHtml(report, opts) {
    opts = opts || {};
    const r = report || {};
    const tr = typeof opts.t === "function" ? opts.t : defaultT;
    const esc = escapeHtml;
    const source = r.source === "llm" ? tr("report_source_llm") : tr("report_source_template");
    const sourceLabel = r.model ? `${source} · ${r.model}` : source;
    const paragraphs = (r.paragraphs || [])
      .map((p) => `<p>${esc(p)}</p>`)
      .join("");
    const timeline = (r.timeline || [])
      .map(
        (e) =>
          `<li><span class="tl-turn">T${(e.turn || 0) + 1}</span> ${esc(e.text || "")}</li>`
      )
      .join("");
    return (
      `<article class="newsreport">` +
      `<div class="nr-top"><h3>${esc(r.headline || "")}</h3>` +
      `<span class="nr-src">${esc(sourceLabel)}</span></div>` +
      `<div class="nr-dateline">${esc(r.dateline || "")}</div>` +
      (r.lead ? `<p class="nr-lead">${esc(r.lead)}</p>` : "") +
      paragraphs +
      (timeline
        ? `<h4>${esc(tr("report_timeline"))}</h4><ul class="nr-timeline">${timeline}</ul>`
        : "") +
      (r.outcome ? `<p class="nr-outcome">${esc(r.outcome)}</p>` : "") +
      `</article>`
    );
  }

  return {
    PALETTE,
    MAJOR_KINDS,
    escapeHtml,
    formatBroadcast,
    niceCeil,
    makeColorFor,
    buildScoreChartSvg,
    buildReportHtml,
  };
});
