// Submit a job, show queue position and progress in `host`, resolve with the finished job row.
import { api, waitForJob } from "./api.js";
import { el, clear, progressBar, notice, spinner } from "./ui.js";

export async function runJob(kind, datasetId, params, host, signal) {
  clear(host).append(spinner("Submitting"));
  let job;
  try {
    job = await api.createJob(kind, datasetId, params);
  } catch (e) {
    clear(host).append(notice("error", e.message));
    throw e;
  }
  const bar = progressBar(0, job.position > 1 ? `Queued, ${job.position - 1} ahead` : "Queued");
  clear(host).append(bar);
  const done = await waitForJob(job.id, (j) => {
    if (j.status === "queued") bar.replaceWith(Object.assign(progressBar(0, j.queue > 1 ? `Queued, ${j.queue - 1} job(s) ahead` : "Queued"), { id: bar.id }));
    else if (j.status === "running" && j.progress) {
      const nb = progressBar(j.progress.fraction, j.progress.message);
      host.firstChild?.replaceWith(nb);
    }
  }, signal);
  clear(host);
  if (done.status === "error") {
    host.append(notice("error", done.error || "The job failed."));
    throw new Error(done.error || "job failed");
  }
  return done;
}
