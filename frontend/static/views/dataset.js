import { api } from "../api.js";
import { el, clear, step, notice, table, svgIcon, details, button, toast, choices, tabs, codeBlock, score } from "../ui.js";
import { histogram, heatmap, boxPlot, scatterChart, barChart } from "../charts.js";

export async function render(root, [id]) {
  const ds = await api.dataset(id);
  const p = ds.profile;
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("div", { class: "crumb" }, el("a", { href: "#/datasets", text: "Home" }), svgIcon("arrow", 12), el("span", { text: ds.name })),
    el("h1", { text: ds.name }),
    el("p", { text: `${p.rows.toLocaleString()} rows and ${p.cols} columns.` }))));

  // ---- Step 2: understand and clean ----------------------------------------
  const warn = p.notes.filter((n) => n.kind === "warn");
  const numeric = p.columns.filter((c) => c.type === "numeric");
  const cats = p.columns.filter((c) => c.type !== "numeric");
  const missing = p.columns.filter((c) => c.missing > 0).sort((a, b) => b.missing_pct - a.missing_pct);

  const overview = () => el("div", { class: "stack" },
    el("div", { class: "tiles" },
      tile("Rows", p.rows.toLocaleString()), tile("Columns", p.cols), tile("Number columns", numeric.length), tile("Text columns", cats.length),
      tile("Columns with blanks", missing.length), tile("Duplicate rows", p.duplicates || 0)),
    warn.length ? el("div", {}, warn.map((n) => notice("warn", simplify(n.text)))) : notice("ok", "No obvious problems: no ID-like columns, no columns that are mostly blank."),
    p.notes.filter((n) => n.kind === "info").map((n) => notice("info", simplify(n.text))),
    missing.length ? el("div", {}, el("h4", { text: "Blanks per column (%)" }), barChart(missing.map((c) => ({ label: c.name, value: c.missing_pct })), { digits: 1 })) : null,
    details("First rows", table(p.head.columns, p.head.rows, { class: "compact" })));

  const columnsTab = () => el("div", { class: "stack" },
    el("p", { class: "muted small", text: "Histogram: the shape of the values. Box: the middle half of the values is the box, the line is the median, red dots are outliers beyond the whiskers." }),
    el("div", { class: "grid grid-2" }, p.columns.map((c) => columnCard(c))));

  const relationships = () => el("div", { class: "stack" },
    p.correlation ? el("div", {}, el("h4", { text: "Which number columns move together" }), el("p", { class: "muted small", text: "Blue means both go up together. Red means one goes up while the other goes down. Dark squares are strong links; two very dark columns say the same thing twice." }), el("div", { class: "chart-wrap" }, heatmap(p.correlation.columns, p.correlation.matrix))) : notice("info", "Correlation needs at least two number columns and at most 30."),
    (p.pairs || []).length ? el("div", {}, el("h4", { text: "The most related pairs" }), el("div", { class: "grid grid-2" }, p.pairs.map((pr) => el("div", { class: "col-card" }, el("h4", { text: `${pr.x} vs ${pr.y} (r = ${pr.r})` }), scatterChart(pr.points, null, { xlabel: pr.x, ylabel: pr.y, height: 240 }))))) : null);

  const plan = p.cleaning || { steps: [], options: {}, script: "" };
  const cleaning = () => el("div", { class: "stack" },
    el("p", { class: "plain", text: "What the tool recommends for this file, in order. Each step has the reason and the pandas code. The button at the bottom applies the whole plan to the prediction flow." }),
    el("ol", { class: "narration" }, plan.steps.map((st) => el("li", {}, el("div", { class: "narr-step", text: st.step }), el("p", { text: st.why }), codeBlock(st.code)))),
    el("div", { class: "row" }, button("Use this plan and predict a column", { kind: "primary big", icon: "arrow", onclick: () => { try { sessionStorage.setItem(`mlhelp.plan.${id}`, JSON.stringify(plan.options)); } catch {} location.hash = `#/dataset/${id}/supervised`; } })),
    details("The whole cleaning script", codeBlock(plan.script)));

  const plotsTab = () => el("div", { class: "stack" },
    el("p", { class: "plain", text: "The same pictures as on this page, as matplotlib and seaborn code you can run in a notebook." }),
    (p.plots || []).map((pl) => el("div", {}, el("h4", { text: pl.title }), el("p", { class: "small muted", text: pl.what }), codeBlock(pl.code))));

  root.append(step(2, "Understand and clean your data", tabs([
    { label: "Overview", render: overview }, { label: "Every column", render: columnsTab }, { label: "Relationships", render: relationships },
    { label: `Cleaning plan (${plan.steps.length})`, render: cleaning }, { label: "Plot code", render: plotsTab },
  ])));

  // ---- Step 3: what to do ----------------------------------------------------
  root.append(step(3, "What do you want to do?", choices([
    { key: "supervised", icon: "flask", title: "Predict a column", text: "Pick a column and the tool finds the model that predicts it best." },
    { key: "unsupervised", icon: "scatter", title: "Find groups or odd rows", text: "No column to predict. Group similar rows, make a 2D map, or spot unusual rows." },
    { key: "predict", icon: "target", title: "Use a model I trained", text: "Give new rows to a model you already built here." },
  ], (k) => (location.hash = `#/dataset/${id}/${k}`))));

  if (!ds.sample) root.append(el("div", { class: "row" }, button("Delete this file", { kind: "ghost danger", icon: "trash", onclick: async () => { if (!confirm("Delete this file and its models?")) return; await api.deleteDataset(id); toast("Deleted"); location.hash = "#/datasets"; } })));
}

function tile(label, value) { return el("div", { class: "tile" }, el("div", { class: "tile-label", text: label }), el("div", { class: "tile-value", text: String(value) })); }

function simplify(t) {
  return t.replace("Imputation will invent a lot of values; consider dropping these columns.", "Blanks will be filled with guesses. You may want to leave these columns out.")
    .replace("It carries no pattern and will be dropped automatically.", "It is left out automatically.")
    .replace("Expect scores to swing a lot between folds; trust cross-validation, not a single split.", "Results will vary a lot. Treat scores as rough.")
    .replace("A log transform or the robust scaler often helps linear models here. Trees do not care.", "The cleaning plan log-transforms these.")
    .replace("Linear models get unstable with these; Ridge or dropping one helps.", "They say the same thing twice; one of each pair can be dropped.");
}

function columnCard(c) {
  const flags = [];
  if (c.looks_id) flags.push(el("span", { class: "badge warn", text: "looks like an ID" }));
  if (c.looks_categorical) flags.push(el("span", { class: "badge", text: "few different values" }));
  if (c.missing_pct > 30) flags.push(el("span", { class: "badge warn", text: "many blanks" }));
  if (c.skewed) flags.push(el("span", { class: "badge", text: "long tail" }));
  if (c.box?.n_outliers) flags.push(el("span", { class: "badge", text: `${c.box.n_outliers} outliers` }));
  const type = { numeric: "number", text: "text", boolean: "yes/no", datetime: "date" }[c.type] || c.type;
  if (c.type === "datetime") flags.push(el("span", { class: "badge", text: "date" }));
  const stats = c.type === "numeric"
    ? [`average ${f(c.mean)}`, `median ${f(c.median)}`, `smallest ${f(c.min)}`, `largest ${f(c.max)}`, `spread ${f(c.std)}`]
    : (c.top || []).slice(0, 5).map((t) => `${t.value} (${t.count})`);
  return el("div", { class: "col-card" },
    el("div", { class: "row" }, el("strong", { text: c.name }), el("span", { class: "badge accent", text: type })),
    el("div", { class: "stat" }, el("span", { text: `${c.missing_pct}% blank` }), el("span", { text: `${c.unique} different values` })),
    flags.length ? el("div", { class: "chips" }, flags) : null,
    c.hist ? histogram(c.hist, { height: 90, mini: true }) : null,
    c.box ? boxPlot(c.box) : null,
    c.type !== "numeric" && c.top && !c.looks_id && c.type !== "datetime" ? barChart(c.top.map((t) => ({ label: t.value, value: t.count })), { digits: 0 }) : null,
    el("div", { class: "stat" }, stats.map((t) => el("span", { text: t }))));
}

function f(v) { return v === null || v === undefined ? "–" : Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(2); }
