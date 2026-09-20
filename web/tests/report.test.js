const test = require("node:test");
const assert = require("node:assert");
const R = require("../report.js");

const TIMELINE = {
  score_history: [
    { turn: 0, scores: { Germany: 150, France: 150, UK: 150, "Soviet Union": 150 } },
    { turn: 1, scores: { Germany: 320, France: 155, UK: 160, "Soviet Union": 158 } },
    { turn: 2, scores: { Germany: 410, France: 0, UK: 170, "Soviet Union": 165 } },
  ],
  events: [
    { kind: "city_captured", turn: 1, actor: "Germany", city: "Paris", previous: "France" },
    { kind: "faction_eliminated", turn: 2, actor: "France" },
    { kind: "unit_destroyed", turn: 2, actor: "UK", unit: "U4", victim: "France" },
  ],
};

test("formatBroadcast covers major event kinds", () => {
  assert.match(R.formatBroadcast({ kind: "city_captured", actor: "A", city: "B", previous: "C" }), /A captured B from C/);
  assert.match(R.formatBroadcast({ kind: "capital_captured", actor: "A", owner: "B", city: "C" }), /B's capital C/);
  assert.match(R.formatBroadcast({ kind: "capital_annexed", actor: "A", owner: "B", city: "C" }), /annexed B/);
  assert.match(R.formatBroadcast({ kind: "faction_eliminated", actor: "A" }), /A has been eliminated/);
});

test("formatBroadcast accepts a custom translator", () => {
  const t = (key, ...args) => `${key}:${args.join(",")}`;
  assert.strictEqual(
    R.formatBroadcast({ kind: "faction_eliminated", actor: "France" }, t),
    "bc_eliminated:France"
  );
});

test("chart draws one polyline per nation and marks major events only", () => {
  const svg = R.buildScoreChartSvg(TIMELINE, {
    nations: ["Germany", "France", "UK", "Soviet Union"],
    colorFor: R.makeColorFor(),
  });
  assert.match(svg, /<svg/);
  assert.strictEqual((svg.match(/class="chart-line"/g) || []).length, 4);
  // Only the two major events get markers; unit_destroyed is ignored.
  assert.strictEqual((svg.match(/class="chart-marker"/g) || []).length, 2);
  assert.match(svg, /Germany captured Paris/);
});

test("chart handles an empty timeline without throwing", () => {
  const svg = R.buildScoreChartSvg({ score_history: [], events: [] }, { emptyText: "nope" });
  assert.match(svg, /nope/);
});

test("niceCeil rounds up sensibly", () => {
  assert.strictEqual(R.niceCeil(0), 10);
  assert.strictEqual(R.niceCeil(11), 20);
  assert.strictEqual(R.niceCeil(410), 500);
});

test("report HTML escapes untrusted text", () => {
  const html = R.buildReportHtml({
    source: "llm",
    headline: "<script>alert(1)</script>",
    dateline: "T",
    lead: "lead & <b>",
    paragraphs: ["p1"],
    timeline: [{ turn: 0, text: "<img src=x onerror=alert(1)>" }],
    outcome: "done",
  });
  assert.ok(!html.includes("<script>"));
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /AI dispatch/);
  assert.match(html, /Timeline of major events/);
});

test("report source badge includes the model label", () => {
  const html = R.buildReportHtml({
    source: "llm",
    model: "zhipu/glm-4-plus",
    headline: "H",
    dateline: "D",
    lead: "L",
    paragraphs: [],
    timeline: [],
    outcome: "O",
  });
  assert.match(html, /zhipu\/glm-4-plus/);
});

test("escapeHtml covers the dangerous characters", () => {
  assert.strictEqual(
    R.escapeHtml("<a href=\"x\">&'"),
    "&lt;a href=&quot;x&quot;&gt;&amp;&#39;"
  );
});
