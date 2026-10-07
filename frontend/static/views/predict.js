import { api } from "../api.js";
import { el, clear, card, button, notice, table, select, field, codeBlock, svgIcon, timeAgo, spinner } from "../ui.js";
import { tryModelPanel } from "../trymodel.js";

export async function render(root, [id]) {
  const [ds, models] = await Promise.all([api.dataset(id), api.models(id)]);
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("div", { class: "crumb" }, el("a", { href: "#/datasets", text: "Home" }), svgIcon("arrow", 12), el("a", { href: `#/dataset/${id}`, text: ds.name }), svgIcon("arrow", 12), el("span", { text: "Use a model" })),
    el("h1", { text: "Use a model on new rows" }),
    el("p", { text: "Type one row by hand, or upload a file with the same columns." }))));
  if (!models.length) { root.append(notice("info", ["You have not trained a model on this file yet. ", el("a", { href: `#/dataset/${id}/supervised`, text: "Train one first." })])); return; }

  const sel = select(models.map((m) => ({ value: m.id, label: `${m.name} (${m.task}, ${Object.entries(m.metrics || {}).slice(0, 1).map(([k, v]) => `${k} ${v}`).join("")}, ${timeAgo(m.created)})` })), models[0].id, () => refresh());
  const body = el("div", { class: "stack" });
  const out = el("div");
  async function refresh() {
    clear(body); clear(out);
    const m = models.find((x) => x.id === sel.value);
    const panel = await tryModelPanel(m.id, { compact: true });
    const file = el("input", { type: "file", accept: ".csv,.tsv,.xlsx", onchange: async () => {
      if (!file.files[0]) return; clear(out).append(spinner("Predicting file"));
      try { clear(out).append(predictions(await api.predictFile(m.id, file.files[0]), m)); } catch (e) { clear(out).append(notice("error", e.message)); }
    } });
    const code = await api.modelCode(m.id);
    body.append(
      el("div", { class: "row" },
        el("a", { class: "btn", href: `/api/models/${m.id}/download`, download: "" }, svgIcon("download"), el("span", { text: "Download .joblib" })),
        el("a", { class: "btn", href: `/api/models/${m.id}/notebook`, download: "" }, svgIcon("book"), el("span", { text: "Download notebook" }))),
      el("h4", { text: "Type one row" }), panel,
      el("h4", { text: "Or upload a file of rows" }), field("CSV or Excel with the same columns", file, "The first 500 rows are predicted."),
      el("h4", { text: "Use it in your own Python" }), codeBlock(`import joblib\npipe = joblib.load('mlhelp-${m.id}.joblib')\npipe.predict(new_rows_dataframe)\n\n# the estimator inside:\n${code}`));
  }
  root.append(card("Pick a model", [field("Model", sel), body]), out);
  refresh();
}

function predictions(res, m) {
  const rows = res.predictions.map((p, i) => [i + 1, p, res.probabilities ? res.probabilities[i].map((x, j) => `${res.classes[j]}: ${(x * 100).toFixed(1)}%`).join("  ") : ""]);
  return card("Predictions", [res.note ? el("p", { class: "muted small", text: res.note }) : null, table(["#", `Predicted ${m.target}`, res.probabilities ? "Probabilities" : ""], rows, { class: "compact" })]);
}
