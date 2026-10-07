import { api } from "../api.js";
import { el, card, table, timeAgo, notice, details, codeBlock, narration, fullCode } from "../ui.js";

export async function render(root, [jobId]) {
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" }, el("h1", { text: "History" }), el("p", { text: "Every job that ran, newest first. Results are kept until the dataset expires." }))));
  if (jobId) {
    const j = await api.job(jobId);
    root.append(card(`${j.kind} job`, [
      el("p", { class: "muted small", text: `${j.status}, ${timeAgo(j.created)}${j.finished ? `, ${(j.finished - j.started).toFixed(1)} s` : ""}` }),
      j.error ? notice("error", j.error) : null,
      details("Parameters", codeBlock(JSON.stringify(j.params, null, 2), "json")),
      j.result?.narration ? details("Narration", narration(j.result.narration), true) : null,
      fullCode(j.result),
      j.result ? details("Raw result", codeBlock(JSON.stringify(j.result, null, 2).slice(0, 20000), "json")) : null,
      el("a", { href: "#/jobs", text: "All jobs" })]));
    return;
  }
  const [jobs, datasets] = await Promise.all([api.jobs(), api.datasets()]);
  const names = Object.fromEntries(datasets.map((d) => [d.id, d.name]));
  if (!jobs.length) { root.append(el("div", { class: "empty", text: "No jobs yet." })); return; }
  root.append(card(null, table([
    { label: "Kind", get: (j) => el("a", { href: `#/jobs/${j.id}`, text: j.kind }) },
    { label: "Dataset", get: (j) => names[j.dataset_id] ? el("a", { href: `#/dataset/${j.dataset_id}`, text: names[j.dataset_id] }) : el("span", { class: "muted", text: "deleted" }) },
    { label: "Detail", get: (j) => j.params.spec ? (j.params.spec.kind === "single" ? j.params.spec.model : `${j.params.spec.kind}: ${(j.params.spec.members || [j.params.spec.model]).join(", ")}`) : j.params.model || j.params.algorithm || (j.params.target ? `target ${j.params.target}` : "") },
    { label: "Status", get: (j) => el("span", { class: `badge ${j.status === "done" ? "ok" : j.status === "error" ? "err" : "warn"}`, text: j.status }) },
    { label: "When", get: (j) => timeAgo(j.created) },
  ], jobs)));
}
