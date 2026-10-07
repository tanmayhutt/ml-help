import { api, cached } from "../api.js";
import { el, step, button, notice, table, svgIcon, bytes, timeAgo, toast, spinner, clear } from "../ui.js";

export async function render(root) {
  root.append(el("section", { class: "hero" },
    el("h1", { text: "Find the best model for your data" }),
    el("p", { text: "Add a table, pick the column to predict, press one button. Every scikit-learn model is tested, the best ones are combined, and you get a clear winner you can try right away, with the full Python code." }),
    el("div", { class: "hero-steps" },
      el("span", { class: "hero-step" }, el("b", { text: "1" }), "Add your data"),
      el("span", { class: "hero-step" }, el("b", { text: "2" }), "Understand and clean it"),
      el("span", { class: "hero-step" }, el("b", { text: "3" }), "Pick what to predict"),
      el("span", { class: "hero-step" }, el("b", { text: "4" }), "Find the best model"),
      el("span", { class: "hero-step" }, el("b", { text: "5" }), "Try it on new values"))));

  const [health, samplesList, list] = await Promise.all([api.health(), cached("samples"), api.datasets()]);
  const L = health.limits;

  const input = el("input", { type: "file", accept: ".csv,.tsv,.txt,.xlsx" });
  const drop = el("label", { class: "drop", tabindex: 0, onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } } },
    svgIcon("upload", 24), el("div", {}, el("strong", { text: "Drop your file here, or click to choose" }), el("div", { class: "small", text: `CSV or Excel. Up to ${L.max_upload_mb} MB and ${L.max_rows.toLocaleString()} rows.` })), input);
  const status = el("div");
  async function handle(file) {
    if (!file) return;
    clear(status).append(spinner(`Reading ${file.name}`));
    try { const ds = await api.upload(file); toast(`Loaded ${ds.rows} rows`); location.hash = `#/dataset/${ds.id}`; }
    catch (e) { clear(status).append(notice("error", e.message)); }
  }
  input.addEventListener("change", () => handle(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => handle(e.dataTransfer.files[0]));

  const samples = el("div", { class: "sample-list" }, samplesList.map((s) => el("div", { class: "sample" },
    el("strong", { text: s.name }), el("span", { class: "badge accent", text: s.task === "classification" ? "predict a category" : "predict a number" }), el("p", { text: s.about }),
    el("div", { class: "row" }, button("Use this", { kind: "small", icon: "arrow", onclick: async (ev) => {
      ev.target.closest("button").disabled = true;
      try { const ds = await api.loadSample(s.key); location.hash = `#/dataset/${ds.id}`; } catch (e) { toast(e.message, "error"); ev.target.closest("button").disabled = false; }
    } })))));

  root.append(step(1, "Add your data", [
    el("p", { class: "plain", text: "Upload a table where each row is one example and each column is one piece of information." }),
    drop, status,
    el("h4", { text: "No file yet? Try an example" }), samples]));

  if (list.length) {
    root.append(step("", "Your files", table(
      [{ label: "Name", get: (r) => el("a", { href: `#/dataset/${r.id}`, text: r.name }) }, { label: "Rows", key: "rows", num: true }, { label: "Columns", key: "cols", num: true },
       { label: "Size", get: (r) => bytes(r.bytes) }, { label: "Added", get: (r) => timeAgo(r.created) },
       { label: "", get: (r) => r.sample ? el("span", { class: "badge", text: "example" }) : button("Delete", { kind: "ghost small danger", icon: "trash", onclick: async (ev) => { ev.stopPropagation(); if (!confirm(`Delete ${r.name}?`)) return; await api.deleteDataset(r.id); render(clear(root)); } }) }],
      list)));
  }
}
