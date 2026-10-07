import { api } from "../api.js";
import { el, clear, card, button, notice, table, select, field, codeBlock, svgIcon, timeAgo, spinner } from "../ui.js";

export async function render(root, [id]) {
  const [ds, models] = await Promise.all([api.dataset(id), api.models(id)]);
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("div", { class: "crumb" }, el("a", { href: "#/datasets", text: "Datasets" }), svgIcon("arrow", 12), el("a", { href: `#/dataset/${id}`, text: ds.name }), svgIcon("arrow", 12), el("span", { text: "Predict" })),
    el("h1", { text: "Predict with a saved model" }),
    el("p", { text: "Every model you train on this dataset is kept for the retention window. Type a row or upload a file of new rows." }))));
  if (!models.length) { root.append(notice("info", ["No models yet. ", el("a", { href: `#/dataset/${id}/supervised`, text: "Train one first." })])); return; }

  const sel = select(models.map((m) => ({ value: m.id, label: `${m.name} (${m.task}, ${Object.entries(m.metrics || {}).slice(0, 1).map(([k, v]) => `${k} ${v}`).join("")}, ${timeAgo(m.created)})` })), models[0].id, () => refresh());
  const body = el("div", { class: "stack" });
  const out = el("div");
  async function refresh() {
    clear(body); clear(out);
    const m = models.find((x) => x.id === sel.value);
    const feats = m.features.features;
    const inputs = {};
    const form = el("div", { class: "grid grid-3" }, feats.map((f) => { const inp = el("input", { type: "text", placeholder: f }); inputs[f] = inp; return field(f, inp); }));
    const runOne = button("Predict this row", { kind: "primary", icon: "target", onclick: async () => {
      const row = {}; for (const f of feats) { const v = inputs[f].value.trim(); row[f] = v === "" ? null : isNaN(Number(v)) ? v : Number(v); }
      clear(out).append(spinner("Predicting"));
      try { clear(out).append(predictions(await api.predict(m.id, [row]), m)); } catch (e) { clear(out).append(notice("error", e.message)); }
    } });
    const file = el("input", { type: "file", accept: ".csv,.tsv,.xlsx", onchange: async () => {
      if (!file.files[0]) return; clear(out).append(spinner("Predicting file"));
      try { clear(out).append(predictions(await api.predictFile(m.id, file.files[0]), m)); } catch (e) { clear(out).append(notice("error", e.message)); }
    } });
    const code = await api.modelCode(m.id);
    body.append(
      el("div", { class: "row" },
        el("a", { class: "btn", href: `/api/models/${m.id}/download`, download: "" }, svgIcon("download"), el("span", { text: "Download .joblib" })),
        el("a", { class: "btn", href: `/api/models/${m.id}/notebook`, download: "" }, svgIcon("book"), el("span", { text: "Download notebook" }))),
      el("h4", { text: "One row" }), form, runOne,
      el("h4", { text: "Or a file of rows" }), field("CSV or XLSX with the same columns", file, "Up to 500 rows are predicted."),
      el("h4", { text: "Use it in Python" }), codeBlock(`import joblib\npipe = joblib.load('mlhelp-${m.id}.joblib')\npipe.predict(new_rows_dataframe)\n\n# the estimator inside:\n${code}`));
  }
  root.append(card("Model", [field("Saved model", sel), body]), out);
  refresh();
}

function predictions(res, m) {
  const rows = res.predictions.map((p, i) => [i + 1, p, res.probabilities ? res.probabilities[i].map((x, j) => `${res.classes[j]}: ${(x * 100).toFixed(1)}%`).join("  ") : ""]);
  return card("Predictions", [res.note ? el("p", { class: "muted small", text: res.note }) : null, table(["#", `Predicted ${m.target}`, res.probabilities ? "Probabilities" : ""], rows, { class: "compact" })]);
}
