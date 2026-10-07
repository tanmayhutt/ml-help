import { api } from "../api.js";
import { el, step, notice, table, svgIcon, bytes, details, button, toast, choices } from "../ui.js";
import { histogram, heatmap } from "../charts.js";

export async function render(root, [id]) {
  const ds = await api.dataset(id);
  const p = ds.profile;
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("div", { class: "crumb" }, el("a", { href: "#/datasets", text: "Home" }), svgIcon("arrow", 12), el("span", { text: ds.name })),
    el("h1", { text: ds.name }),
    el("p", { text: `${p.rows.toLocaleString()} rows and ${p.cols} columns.` }))));

  root.append(step(2, "What do you want to do?", choices([
    { key: "supervised", icon: "flask", title: "Predict a column", text: "Pick a column and the tool finds the model that predicts it best." },
    { key: "unsupervised", icon: "scatter", title: "Find groups or odd rows", text: "No column to predict. Group similar rows, make a 2D map, or spot unusual rows." },
    { key: "predict", icon: "target", title: "Use a model I trained", text: "Give new rows to a model you already built here." },
  ], (k) => (location.hash = `#/dataset/${id}/${k}`))));

  const warn = p.notes.filter((n) => n.kind === "warn");
  if (warn.length) root.append(step("", "Things to know about this file", warn.map((n) => notice("warn", simplify(n.text)))));

  root.append(step("", "A look at your data", [
    details("First rows", table(p.head.columns, p.head.rows, { class: "compact" }), true),
    details("Each column", el("div", { class: "grid grid-2" }, p.columns.map((c) => columnCard(c)))),
    p.correlation ? details("Which number columns move together", [el("p", { class: "muted small", text: "Blue means both go up together. Red means one goes up while the other goes down. Dark squares are strong links." }), el("div", { class: "chart-wrap" }, heatmap(p.correlation.columns, p.correlation.matrix))]) : null,
  ]));
  if (!ds.sample) root.append(el("div", { class: "row" }, button("Delete this file", { kind: "ghost danger", icon: "trash", onclick: async () => { if (!confirm("Delete this file and its models?")) return; await api.deleteDataset(id); toast("Deleted"); location.hash = "#/datasets"; } })));
}

function simplify(t) {
  return t.replace("Imputation will invent a lot of values; consider dropping these columns.", "Blanks will be filled with guesses. You may want to leave these columns out.")
    .replace("It carries no pattern and will be dropped automatically.", "It is left out automatically.")
    .replace("Expect scores to swing a lot between folds; trust cross-validation, not a single split.", "Results will vary a lot. Treat scores as rough.");
}

function columnCard(c) {
  const flags = [];
  if (c.looks_id) flags.push(el("span", { class: "badge warn", text: "looks like an ID" }));
  if (c.looks_categorical) flags.push(el("span", { class: "badge", text: "few different values" }));
  if (c.missing_pct > 30) flags.push(el("span", { class: "badge warn", text: "many blanks" }));
  const type = { numeric: "number", text: "text", boolean: "yes/no", datetime: "date" }[c.type] || c.type;
  const stats = c.type === "numeric"
    ? [`average ${f(c.mean)}`, `smallest ${f(c.min)}`, `largest ${f(c.max)}`]
    : (c.top || []).slice(0, 4).map((t) => `${t.value} (${t.count})`);
  return el("div", { class: "col-card" },
    el("div", { class: "row" }, el("strong", { text: c.name }), el("span", { class: "badge accent", text: type })),
    el("div", { class: "stat" }, el("span", { text: `${c.missing_pct}% blank` }), el("span", { text: `${c.unique} different values` })),
    flags.length ? el("div", { class: "chips" }, flags) : null,
    c.hist ? histogram(c.hist, { height: 100 }) : null,
    el("div", { class: "stat" }, stats.map((t) => el("span", { text: t }))));
}

function f(v) { return v === null || v === undefined ? "–" : Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(2); }
