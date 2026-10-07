import { api } from "../api.js";
import { el, card, notice, table, svgIcon, bytes, details, button, toast } from "../ui.js";
import { histogram, heatmap } from "../charts.js";

export async function render(root, [id]) {
  const ds = await api.dataset(id);
  const p = ds.profile;
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("div", { class: "crumb" }, el("a", { href: "#/datasets", text: "Datasets" }), svgIcon("arrow", 12), el("span", { text: ds.name })),
    el("h1", { text: ds.name }),
    el("p", { text: `${p.rows.toLocaleString()} rows, ${p.cols} columns, ${bytes(p.memory_bytes)} in memory.` }))));

  root.append(el("div", { class: "launch" },
    launch(`#/dataset/${id}/supervised`, "flask", "Supervised: predict a column", "Pick a target, race every model, build ensembles, tune the winner."),
    launch(`#/dataset/${id}/unsupervised`, "scatter", "Unsupervised: find structure", "Cluster, reduce dimensions, and flag anomalies without a target."),
    launch(`#/dataset/${id}/predict`, "target", "Predict with a saved model", "Use a model you trained here on new rows."),
  ));

  if (p.notes.length) root.append(card("What the profiler noticed", p.notes.map((n) => notice(n.kind === "warn" ? "warn" : "info", n.text))));

  const colsBody = el("div", { class: "grid grid-2" }, p.columns.map((c) => columnCard(c)));
  root.append(card("Columns", [el("p", { class: "muted small", text: "Type, missing share, distinct count, and a histogram for numbers. The type decides how preprocessing treats the column: numbers get imputed and scaled, text gets imputed and encoded." }), colsBody]));

  root.append(card("First rows", table(p.head.columns, p.head.rows, { class: "compact" })));

  if (p.correlation) {
    root.append(card("Correlation between numeric columns", [
      el("p", { class: "muted small", text: "Pearson correlation from -1 (blue is positive, red is negative). Strong pairs are redundant: linear models become unstable with them, and one of the pair can usually be dropped." }),
      el("div", { class: "chart-wrap" }, heatmap(p.correlation.columns, p.correlation.matrix))]));
  }
  if (!ds.sample) root.append(el("div", { class: "row" }, button("Delete dataset", { kind: "ghost danger", icon: "trash", onclick: async () => { if (!confirm("Delete this dataset and its models?")) return; await api.deleteDataset(id); toast("Deleted"); location.hash = "#/datasets"; } })));
}

function launch(href, icon, title, text) {
  return el("a", { href }, el("strong", {}, svgIcon(icon, 18), title), el("span", { text }));
}

function columnCard(c) {
  const flags = [];
  if (c.looks_id) flags.push(el("span", { class: "badge warn", text: "looks like an ID" }));
  if (c.looks_categorical) flags.push(el("span", { class: "badge", text: "few distinct values" }));
  if (c.missing_pct > 30) flags.push(el("span", { class: "badge warn", text: "many missing" }));
  if (c.outliers) flags.push(el("span", { class: "badge", text: `${c.outliers} outliers` }));
  const stats = c.type === "numeric"
    ? [`mean ${f(c.mean)}`, `std ${f(c.std)}`, `min ${f(c.min)}`, `median ${f(c.median)}`, `max ${f(c.max)}`, c.skew !== null && c.skew !== undefined ? `skew ${f(c.skew)}` : null]
    : (c.top || []).map((t) => `${t.value} (${t.count})`);
  return el("div", { class: "col-card" },
    el("div", { class: "row" }, el("strong", { text: c.name }), el("span", { class: "badge accent", text: c.type })),
    el("div", { class: "stat" }, el("span", { text: `${c.missing_pct}% missing` }), el("span", { text: `${c.unique} distinct` })),
    flags.length ? el("div", { class: "chips" }, flags) : null,
    c.hist ? histogram(c.hist, { height: 110 }) : null,
    el("div", { class: "stat" }, stats.filter(Boolean).map((t) => el("span", { text: t }))));
}

function f(v) { return v === null || v === undefined ? "–" : Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(3); }
