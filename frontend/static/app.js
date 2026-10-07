// Hash router and app shell.
import { api, ApiError, setToken } from "./api.js";
import { el, clear, notice, button, field } from "./ui.js";
import * as datasets from "./views/datasets.js";
import * as dataset from "./views/dataset.js";
import * as experiment from "./views/experiment.js";
import * as unsupervised from "./views/unsupervised.js";
import * as predict from "./views/predict.js";
import * as learn from "./views/learn.js";
import * as jobs from "./views/jobs.js";

const routes = [
  { pattern: /^\/?$/, view: datasets, nav: "datasets" },
  { pattern: /^\/datasets$/, view: datasets, nav: "datasets" },
  { pattern: /^\/dataset\/([^/]+)$/, view: dataset, nav: "datasets" },
  { pattern: /^\/dataset\/([^/]+)\/supervised$/, view: experiment, nav: "datasets" },
  { pattern: /^\/dataset\/([^/]+)\/unsupervised$/, view: unsupervised, nav: "datasets" },
  { pattern: /^\/dataset\/([^/]+)\/predict$/, view: predict, nav: "datasets" },
  { pattern: /^\/learn(?:\/([^/]+))?$/, view: learn, nav: "learn" },
  { pattern: /^\/jobs(?:\/([^/]+))?$/, view: jobs, nav: "jobs" },
];

const main = document.getElementById("main");
let currentAbort = null;

async function render() {
  const path = location.hash.replace(/^#/, "") || "/";
  const route = routes.find((r) => r.pattern.test(path));
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === route?.nav));
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
    clear(slot).append(el("span", { text: h.queue ? `${h.queue} job${h.queue > 1 ? "s" : ""} in queue` : "idle" }));
  } catch { /* leave empty */ }
}

window.addEventListener("hashchange", render);
render();
status();
