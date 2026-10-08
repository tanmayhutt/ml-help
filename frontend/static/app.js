// Hash router and app shell.
import { api, ApiError, setToken } from "./api.js";
import { el, clear, notice, button, field } from "./ui.js";
import * as datasets from "./views/datasets.js";
import * as dataset from "./views/dataset.js";
import * as experiment from "./views/experiment.js";
import * as unsupervised from "./views/unsupervised.js";
import * as predict from "./views/predict.js";
import * as jobs from "./views/jobs.js";
import * as model from "./views/model.js";

const routes = [
  { pattern: /^\/?$/, view: datasets, nav: "datasets" },
  { pattern: /^\/datasets$/, view: datasets, nav: "datasets" },
  { pattern: /^\/dataset\/([^/]+)$/, view: dataset, nav: "datasets" },
  { pattern: /^\/dataset\/([^/]+)\/supervised$/, view: experiment, nav: "datasets" },
  { pattern: /^\/dataset\/([^/]+)\/unsupervised$/, view: unsupervised, nav: "datasets" },
  { pattern: /^\/dataset\/([^/]+)\/predict$/, view: predict, nav: "datasets" },
  { pattern: /^\/jobs(?:\/([^/]+))?$/, view: jobs, nav: "jobs" },
  { pattern: /^\/model\/([^/]+)$/, view: model, nav: "datasets" },
];

const main = document.getElementById("main");
const JOURNEY = [
  { key: "data", label: "Add your data", href: () => "#/datasets" },
  { key: "understand", label: "Understand and clean", href: (id) => (id ? `#/dataset/${id}` : "#/datasets") },
  { key: "predict", label: "Pick what to predict", href: (id) => (id ? `#/dataset/${id}/supervised` : null) },
  { key: "result", label: "Find the best model", href: (id) => (id ? `#/dataset/${id}/supervised` : null) },
  { key: "try", label: "Try it on new values", href: (id) => (id ? `#/dataset/${id}/predict` : null) },
];
function journey(path) {
  const ol = document.getElementById("journey");
  if (!ol) return;
  const m = path.match(/^\/dataset\/([^/]+)(?:\/([^/]+))?/);
  const id = m ? m[1] : null; const sub = m ? m[2] : null;
  let now = 0;
  if (m && !sub) now = 1; else if (sub === "supervised" || sub === "unsupervised") now = 2; else if (sub === "predict" || path.startsWith("/model/")) now = 4;
  if (path.startsWith("/jobs")) now = -1;
  ol.replaceChildren(...JOURNEY.map((st, i) => {
    const href = st.href(id);
    const li = document.createElement("li");
    li.className = i < now ? "done" : i === now ? "now" : "";
    if (href && i <= Math.max(now, 1)) { const a = document.createElement("a"); a.href = href; a.textContent = st.label; li.append(a); } else li.textContent = st.label;
    return li;
  }));
}
let currentAbort = null;

async function render() {
  const path = location.hash.replace(/^#/, "") || "/";
  const route = routes.find((r) => r.pattern.test(path));
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === route?.nav));
  journey(path);
  if (currentAbort) currentAbort.abort();
  currentAbort = new AbortController();
  clear(main);
  if (!route) { main.append(el("div", { class: "empty" }, el("h2", { text: "Page not found" }), el("a", { href: "#/datasets", text: "Back to datasets" }))); return; }
  const params = path.match(route.pattern).slice(1);
  try {
    await route.view.render(main, params, currentAbort.signal);
    window.scrollTo(0, 0);
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) return showTokenPrompt();
    if (e.message === "cancelled") return;
    console.error(e);
    main.append(notice("error", e.message || "Something went wrong."));
  }
}

function showTokenPrompt() {
  clear(main);
  const input = el("input", { type: "password", autocomplete: "off", placeholder: "Access token" });
  const form = el("form", { class: "card", onsubmit: (e) => { e.preventDefault(); setToken(input.value.trim()); render(); } },
    el("div", { class: "card-body stack" },
      el("h2", { text: "Access token required" }),
      el("p", { class: "muted", text: "This instance is private. Enter the token the owner gave you." }),
      field("Token", input), button("Continue", { kind: "primary", type: "submit" })));
  main.append(form);
  input.focus();
}

async function status() {
  const slot = document.getElementById("status-slot");
  try {
    const h = await api.health();
    clear(slot).append(el("span", { text: h.queue ? `${h.queue} job${h.queue > 1 ? "s" : ""} in queue` : "Server ready" }));
  } catch { /* leave empty */ }
}

window.addEventListener("hashchange", render);
render();
status();
