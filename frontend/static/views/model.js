import { api } from "../api.js";
import { el, notice, svgIcon, score, details, codeBlock, METRIC_NAMES } from "../ui.js";
import { tryModelPanel } from "../trymodel.js";

export async function render(root, [id]) {
  let m;
  try { m = await api.model(id); } catch (e) { root.append(notice("error", e.message)); return; }
  const metric = Object.entries(m.metrics || {})[0];
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("div", { class: "crumb" }, el("a", { href: "#/datasets", text: "Home" }), svgIcon("arrow", 12), m.dataset_name ? el("a", { href: `#/dataset/${m.dataset_id}`, text: m.dataset_name }) : null, svgIcon("arrow", 12), el("span", { text: "Model" })),
    el("h1", { text: m.name }),
    el("p", { text: `Predicts ${m.target}${metric ? `. ${METRIC_NAMES[metric[0]] || metric[0]} ${score(metric[1])} on rows it never saw.` : "."}` }))));
  root.append(el("section", { class: "card" }, el("div", { class: "card-body" }, await tryModelPanel(id, { autorun: true }))));
  const code = await api.modelCode(id).catch(() => "");
  root.append(el("section", { class: "card" }, el("div", { class: "card-body stack" },
    el("div", { class: "row" },
      el("a", { class: "btn", href: `/api/models/${m.id}/download`, download: "" }, svgIcon("download"), el("span", { text: "Download model (.joblib)" })),
      el("a", { class: "btn", href: `/api/models/${m.id}/notebook`, download: "" }, svgIcon("book"), el("span", { text: "Download notebook" }))),
    details("Use it in your own Python", codeBlock(`import joblib\npipe = joblib.load('mlhelp-${m.id}.joblib')\npipe.predict(new_rows_dataframe)\n\n# the estimator inside:\n${code}`)))));
}
