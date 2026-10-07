// "Try the model" panel: typed inputs built from the model's input schema, instant prediction, shareable link.
import { api } from "./api.js";
import { el, clear, button, notice, field, select, svgIcon, score, codeBlock, spinner, toast } from "./ui.js";

export async function tryModelPanel(modelId, opts = {}) {
  let m;
  try { m = await api.model(modelId); } catch (e) { return notice("error", e.message); }
  if (!m.file_exists) return notice("warn", "This model's file has expired. Train it again to use it.");
  const inputs = {};
  const form = el("div", { class: "try-grid" }, m.inputs.map((inp) => {
    let ctl;
    if (inp.kind === "number") ctl = el("input", { type: "number", step: "any", value: inp.default ?? "", placeholder: inp.min !== undefined ? `${inp.min} to ${inp.max}` : "" });
    else if (inp.kind === "choice") ctl = select(inp.choices.map((c) => ({ value: c, label: c })), inp.default);
    else ctl = el("input", { type: "text", value: inp.default || "" });
    inputs[inp.name] = ctl;
    return field(inp.name, ctl, inp.kind === "number" && inp.min !== undefined ? `typical range ${fmt(inp.min)} to ${fmt(inp.max)}` : null);
  }));
  const out = el("div", { class: "try-out" });
  async function predict() {
    const row = {};
    for (const inp of m.inputs) { const v = inputs[inp.name].value; row[inp.name] = v === "" ? null : inp.kind === "number" ? Number(v) : v; }
    clear(out).append(spinner("Predicting"));
    try {
      const res = await api.predict(m.id, [row]);
      clear(out).append(answer(res, m));
    } catch (e) { clear(out).append(notice("error", e.message)); }
  }
  const run = button(`Predict ${m.target}`, { kind: "primary big", icon: "target", onclick: predict });
  const shareUrl = `${location.origin}/#/model/${m.id}`;
  const share = button("Copy link to this model", { kind: "ghost small", icon: "code", onclick: () => { navigator.clipboard?.writeText(shareUrl); toast("Link copied"); } });
  const panel = el("div", { class: "try-panel" },
    opts.compact ? null : el("div", { class: "try-head" }, el("div", {}, el("h3", { text: `Try ${m.name}` }), el("p", { class: "muted small", text: `Trained on ${m.dataset_name || "your file"} to predict ${m.target}. Change any value and press predict.` }))),
    form, el("div", { class: "row" }, run, opts.compact ? el("a", { class: "btn", href: `#/model/${m.id}` }, svgIcon("arrow"), el("span", { text: "Open as its own page" })) : null, share), out);
  if (opts.autorun) setTimeout(predict, 0);
  return panel;
}

function answer(res, m) {
  const pred = res.predictions[0];
  const wrap = el("div", { class: "answer" }, el("div", { class: "answer-label", text: `Predicted ${m.target}` }), el("div", { class: "answer-value", text: String(pred) }));
  if (res.probabilities) {
    const probs = res.probabilities[0];
    wrap.append(el("div", { class: "prob-list" }, res.classes.map((c, i) => el("div", { class: "prob-row" + (String(c) === String(pred) ? " on" : "") },
      el("span", { class: "prob-name", text: c }), el("div", { class: "prob-track" }, el("div", { class: "prob-bar", style: { width: `${Math.round(probs[i] * 100)}%` } })), el("span", { class: "prob-pct", text: `${(probs[i] * 100).toFixed(1)}%` })))));
    wrap.append(el("p", { class: "muted small", text: res.threshold ? `The bars are how confident the model is in each answer. Because '${res.classes[1]}' is the rare class, it is called whenever its confidence is at least ${Math.round(res.threshold * 100)}%, not 50%.` : "The bars are how confident the model is in each answer. Confidence near 50/50 means the model is unsure for this row." }));
  }
  return wrap;
}

function fmt(v) { return v === null || v === undefined ? "" : Math.abs(v) >= 1000 ? Math.round(v) : Number(v.toFixed(2)); }
