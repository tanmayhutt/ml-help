import { api } from "../api.js";
import { el, card, table, timeAgo, notice, details, codeBlock, narration, fullCode } from "../ui.js";
const KIND = { leaderboard: "find best model", train: "train a model", tune: "fine-tune", curve: "more data check", cluster: "find groups", reduce: "2D picture", anomaly: "unusual rows" };

export async function render(root, [jobId]) {
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" }, el("h1", { text: "History" }), el("p", { text: "Everything you have run, newest first." }))));
  if (jobId) {
    const j = await api.job(jobId);
    root.append(card(KIND[j.kind] || j.kind, [
      el("p", { class: "muted small", text: `${j.status}, ${timeAgo(j.created)}${j.finished ? `, ${(j.finished - j.started).toFixed(1)} s` : ""}` }),
      j.error ? notice("error", j.error) : null,
      details("Settings used", codeBlock(JSON.stringify(j.params, null, 2), "json")),
      j.result?.narration ? details("How this was done, step by step", narration(j.result.narration), true) : null,
      fullCode(j.result),
      j.result ? details("Raw numbers", codeBlock(JSON.stringify(j.result, null, 2).slice(0, 20000), "json")) : null,
      el("a", { href: "#/jobs", text: "All jobs" })]));
    return;
  }
  const [jobs, datasets] = await Promise.all([api.jobs(), api.datasets()]);
  const names = Object.fromEntries(datasets.map((d) => [d.id, d.name]));
  if (!jobs.length) { root.append(el("div", { class: "empty", text: "No jobs yet." })); return; }
  root.append(card(null, table([
    { label: "What", get: (j) => el("a", { href: `#/jobs/${j.id}`, text: KIND[j.kind] || j.kind }) },
    { label: "File", get: (j) => names[j.dataset_id] ? el("a", { href: `#/dataset/${j.dataset_id}`, text: names[j.dataset_id] }) : el("span", { class: "muted", text: "deleted" }) },
    { label: "Detail", get: (j) => j.params.spec ? (j.params.spec.kind === "single" ? j.params.spec.model : `${j.params.spec.kind}: ${(j.params.spec.members || [j.params.spec.model]).join(", ")}`) : j.params.model || j.params.algorithm || (j.params.target ? `target ${j.params.target}` : "") },
    { label: "Status", get: (j) => el("span", { class: `badge ${j.status === "done" ? "ok" : j.status === "error" ? "err" : "warn"}`, text: { done: "finished", error: "failed", running: "running", queued: "waiting" }[j.status] || j.status }) },
    { label: "When", get: (j) => timeAgo(j.created) },
  ], jobs)));
}
