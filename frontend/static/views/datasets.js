import { api, cached } from "../api.js";
import { el, card, button, notice, table, svgIcon, bytes, timeAgo, toast, spinner, clear } from "../ui.js";

export async function render(root) {
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("h1", { text: "Datasets" }),
    el("p", { text: "Upload a CSV or Excel file, or start with a built-in teaching dataset. The next screen profiles the data and lets you launch experiments." }))));

  const [health, samplesList, list] = await Promise.all([api.health(), cached("samples"), api.datasets()]);
  const L = health.limits;

  // Upload
  const input = el("input", { type: "file", accept: ".csv,.tsv,.txt,.xlsx" });
  const dropText = el("div", {}, el("strong", { text: "Drop a file here or click to choose" }), el("div", { class: "small", text: `CSV, TSV, or XLSX up to ${L.max_upload_mb} MB, ${L.max_rows.toLocaleString()} rows, ${L.max_cols} columns.` }));
  const drop = el("label", { class: "drop", tabindex: 0, onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } } }, svgIcon("upload", 24), dropText, input);
  const status = el("div");
  async function handle(file) {
    if (!file) return;
    clear(status).append(spinner(`Uploading and profiling ${file.name}`));
    try {
      const ds = await api.upload(file);
      toast(`Loaded ${ds.name}: ${ds.rows} rows`);
      location.hash = `#/dataset/${ds.id}`;
    } catch (e) {
      clear(status).append(notice("error", e.message));
    }
  }
  input.addEventListener("change", () => handle(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => handle(e.dataTransfer.files[0]));
  root.append(card("Upload your data", [drop, status]));

  // Samples
  const grid = el("div", { class: "sample-list" }, samplesList.map((s) => el("div", { class: "sample" },
    el("strong", { text: s.name }), el("span", { class: "badge accent", text: s.task }), el("p", { text: s.about }),
    el("div", { class: "row" }, button("Open", { kind: "small", icon: "arrow", onclick: async (ev) => {
      ev.target.closest("button").disabled = true;
      try { const ds = await api.loadSample(s.key); location.hash = `#/dataset/${ds.id}`; } catch (e) { toast(e.message, "error"); ev.target.closest("button").disabled = false; }
    } })))));
  root.append(card("Teaching datasets", [el("p", { class: "muted small", text: "Bundled with scikit-learn, no download. Good for lessons and for learning what each chart means before using your own data." }), grid]));

  // Existing
  if (list.length) {
    root.append(card("Loaded datasets", table(
      [{ label: "Name", get: (r) => el("a", { href: `#/dataset/${r.id}`, text: r.name }) }, { label: "Rows", key: "rows", num: true }, { label: "Columns", key: "cols", num: true },
       { label: "Size", get: (r) => bytes(r.bytes) }, { label: "Loaded", get: (r) => timeAgo(r.created) },
       { label: "", get: (r) => r.sample ? el("span", { class: "badge", text: "sample" }) : button("Delete", { kind: "ghost small danger", icon: "trash", onclick: async (ev) => { ev.stopPropagation(); if (!confirm(`Delete ${r.name} and its models?`)) return; await api.deleteDataset(r.id); render(clear(root)); } }) }],
      list)));
  }
  root.append(notice("info", `Server limits: one job at a time, ${L.job_timeout_sec} s per job, model races use up to ${L.leaderboard_rows.toLocaleString()} rows, data is kept ${L.retention_days} days.`));
}
