// DOM helpers and shared components. No framework, just functions that return elements.

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  append(node, children);
  return node;
}

export function append(node, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

export function svgIcon(name, size = 16) {
  const paths = ICONS[name] || ICONS.dot;
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24"); s.setAttribute("width", size); s.setAttribute("height", size);
  s.setAttribute("fill", "none"); s.setAttribute("stroke", "currentColor"); s.setAttribute("stroke-width", "1.8");
  s.setAttribute("stroke-linecap", "round"); s.setAttribute("stroke-linejoin", "round"); s.setAttribute("aria-hidden", "true");
  s.classList.add("icon");
  s.innerHTML = paths;
  return s;
}

const ICONS = {
  dot: '<circle cx="12" cy="12" r="2"/>',
  database: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
  flask: '<path d="M9 3h6"/><path d="M10 3v6L4.5 19a1 1 0 0 0 .9 1.5h13.2a1 1 0 0 0 .9-1.5L14 9V3"/>',
  layers: '<path d="m12 2 9 5-9 5-9-5 9-5z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
  book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
  list: '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>',
  play: '<path d="m6 4 14 8-14 8V4z"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
  warn: '<path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
  arrow: '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
  back: '<path d="M19 12H5"/><path d="m12 19-7-7 7-7"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
  target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
  trash: '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
  spinner: '<path d="M21 12a9 9 0 1 1-6.2-8.6"/>',
  code: '<path d="m16 18 6-6-6-6"/><path d="m8 6-6 6 6 6"/>',
  sliders: '<path d="M4 21v-7"/><path d="M4 10V3"/><path d="M12 21v-9"/><path d="M12 8V3"/><path d="M20 21v-5"/><path d="M20 12V3"/><path d="M2 14h4"/><path d="M10 8h4"/><path d="M18 16h4"/>',
  wand: '<path d="m15 4 1 1"/><path d="m20 9 1 1"/><path d="m3 21 12-12"/><path d="m15 9 6-6"/><path d="M4 7h.01"/><path d="M9 3h.01"/><path d="M20 15h.01"/>',
  scatter: '<circle cx="7" cy="14" r="1.5"/><circle cx="11" cy="8" r="1.5"/><circle cx="16" cy="11" r="1.5"/><circle cx="18" cy="5" r="1.5"/><path d="M3 3v18h18"/>',
};

export function button(label, opts = {}) {
  const b = el("button", { class: "btn " + (opts.kind || ""), type: opts.type || "button", onclick: opts.onclick, disabled: opts.disabled, title: opts.title });
  if (opts.icon) b.append(svgIcon(opts.icon));
  b.append(el("span", { text: label }));
  return b;
}

export function card(title, body, opts = {}) {
  return el("section", { class: "card " + (opts.class || "") },
    title ? el("header", { class: "card-head" }, el("h3", { text: title }), opts.aside || null) : null,
    el("div", { class: "card-body" }, body));
}

export function notice(kind, text) {
  return el("div", { class: `notice notice-${kind}`, role: kind === "error" ? "alert" : "status" }, svgIcon(kind === "info" ? "info" : kind === "ok" ? "check" : "warn"), el("div", {}, text));
}

export function codeBlock(code, lang = "python") {
  const pre = el("pre", { class: "code", dataset: { lang } }, el("code", { text: code }));
  const copy = button("Copy", { kind: "ghost small", icon: "code", onclick: () => { navigator.clipboard?.writeText(code); copy.querySelector("span").textContent = "Copied"; setTimeout(() => (copy.querySelector("span").textContent = "Copy"), 1200); } });
  return el("div", { class: "code-wrap" }, el("div", { class: "code-bar" }, el("span", { class: "code-lang", text: lang }), copy), pre);
}

export function table(columns, rows, opts = {}) {
  const t = el("table", { class: "table " + (opts.class || "") });
  t.append(el("thead", {}, el("tr", {}, columns.map((c) => el("th", { text: typeof c === "string" ? c : c.label, class: c.num ? "num" : "" })))));
  const tb = el("tbody");
  for (const r of rows) {
    tb.append(el("tr", { onclick: opts.onRow ? () => opts.onRow(r) : null, class: opts.onRow ? "clickable" : "" },
      columns.map((c, i) => {
        const v = Array.isArray(r) ? r[i] : typeof c === "string" ? r[c] : c.get ? c.get(r) : r[c.key];
        return el("td", { class: (typeof c !== "string" && c.num) ? "num" : "" }, v instanceof Node ? v : fmt(v));
      })));
  }
  t.append(tb);
  return el("div", { class: "table-scroll" }, t);
}

export function fmt(v, digits = 4) {
  if (v === null || v === undefined) return el("span", { class: "muted", text: "null" });
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(parseFloat(v.toFixed(digits)));
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}

export const METRIC_NAMES = {
  accuracy: "Accuracy", balanced_accuracy: "Balanced accuracy", precision: "Precision", recall: "Recall", f1: "F1 score", mcc: "MCC", roc_auc: "ROC AUC", log_loss: "Log loss",
  r2: "R² score", adjusted_r2: "Adjusted R²", mae: "Average error (MAE)", rmse: "RMSE", mse: "MSE", mape: "Average % error", max_error: "Biggest error",
  silhouette: "Silhouette", davies_bouldin: "Davies-Bouldin", calinski_harabasz: "Calinski-Harabasz",
};

export function metricTiles(metrics, help, primary) {
  const wrap = el("div", { class: "tiles" });
  for (const [k, v] of Object.entries(metrics || {})) {
    if (v === null || v === undefined) continue;
    wrap.append(el("div", { class: "tile " + (k === primary ? "primary" : ""), title: help?.[k] || "" },
      el("div", { class: "tile-label", text: METRIC_NAMES[k] || k.replace(/_/g, " ") }),
      el("div", { class: "tile-value", text: Math.abs(v) >= 1000 ? v.toFixed(1) : v.toFixed(4) })));
  }
  return wrap;
}

export function narration(steps) {
  if (!steps || !steps.length) return null;
  return el("ol", { class: "narration" }, steps.map((s) => el("li", {},
    el("div", { class: "narr-step", text: s.step }),
    el("p", { text: s.text }),
    s.code ? codeBlock(s.code) : null)));
}

export function field(label, input, help) {
  return el("label", { class: "field" }, el("span", { class: "field-label", text: label }), input, help ? el("span", { class: "field-help", text: help }) : null);
}

export function select(options, value, onchange, attrs = {}) {
  const s = el("select", { onchange: (e) => onchange && onchange(e.target.value), ...attrs });
  for (const o of options) {
    const opt = el("option", { value: o.value ?? o, text: o.label ?? o });
    if ((o.value ?? o) == value) opt.selected = true;
    s.append(opt);
  }
  return s;
}

export function spinner(text) {
  return el("div", { class: "spinner-row" }, svgIcon("spinner"), el("span", { text: text || "Working" }));
}

export function progressBar(fraction, label) {
  return el("div", { class: "progress", role: "progressbar", "aria-valuenow": Math.round((fraction || 0) * 100) },
    el("div", { class: "progress-bar", style: { width: `${Math.round((fraction || 0) * 100)}%` } }),
    el("span", { class: "progress-label", text: label || "" }));
}

export function details(summary, body, open = false) {
  const d = el("details", { open: open || null }, el("summary", { text: summary }), el("div", { class: "details-body" }, body));
  return d;
}

export function tabs(items, initial = 0) {
  const nav = el("div", { class: "tabs", role: "tablist" });
  const panel = el("div", { class: "tab-panel" });
  let current = -1;
  const show = (i) => {
    if (i === current) return;
    current = i;
    [...nav.children].forEach((b, j) => { b.classList.toggle("active", i === j); b.setAttribute("aria-selected", i === j); });
    clear(panel).append(items[i].render());
  };
  items.forEach((it, i) => nav.append(el("button", { class: "tab", role: "tab", type: "button", text: it.label, onclick: () => show(i) })));
  show(initial);
  return el("div", { class: "tabs-wrap" }, nav, panel);
}

export function timeAgo(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

export function bytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}

export function toast(text, kind = "info") {
  let host = document.getElementById("toasts");
  if (!host) { host = el("div", { id: "toasts" }); document.body.append(host); }
  const t = el("div", { class: `toast toast-${kind}`, text });
  host.append(t);
  setTimeout(() => t.remove(), 4000);
}

// Full reproducible script attached to every job result. Open by default: reading the code is the point.
export function fullCode(r) {
  if (!r || !r.code) return null;
  return details("Full Python code for this experiment", [el("p", { class: "muted small", text: "The exact pipeline the server ran, as one script. Copy it into a notebook, point it at your CSV, and run it top to bottom." }), codeBlock(r.code)], true);
}

// Numbered step card. `done` marks finished steps; `locked` greys out steps that are not reachable yet.
export function step(n, title, body, opts = {}) {
  return el("section", { class: "card step " + (opts.locked ? "locked" : "") + (opts.done ? " done" : ""), id: opts.id || null },
    el("header", { class: "card-head" }, n !== "" && n !== null && n !== undefined ? el("span", { class: "step-n", text: n }) : null, el("h3", { text: title }), opts.aside || null),
    el("div", { class: "card-body" }, opts.locked ? el("p", { class: "muted", text: opts.lockedText || "Finish the step above first." }) : body));
}

// Big choice buttons for "what do you want to do" screens.
export function choices(items, onpick) {
  const wrap = el("div", { class: "choices" });
  items.forEach((it) => wrap.append(el("button", { type: "button", class: "choice", onclick: () => { [...wrap.children].forEach((c) => c.classList.remove("on")); event.currentTarget.classList.add("on"); onpick(it.key); } },
    el("strong", {}, svgIcon(it.icon || "arrow", 18), it.title), el("span", { text: it.text }))));
  return wrap;
}

// Scores shown to people: 3 decimals, no trailing noise.
export function score(v) { return v === null || v === undefined ? "–" : Number(v).toFixed(3); }
