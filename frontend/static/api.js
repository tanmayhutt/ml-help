// Thin fetch wrapper. Every call goes through here so the access token and error shape stay in one place.
const TOKEN_KEY = "mlhelp.token";

export function getToken() { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; } }
export function setToken(t) { try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch {} }

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

async function request(method, path, body, isForm) {
  const headers = {};
  const token = getToken();
  if (token) headers["X-Access-Token"] = token;
  if (body && !isForm) headers["Content-Type"] = "application/json";
  const res = await fetch("/api" + path, { method, headers, body: isForm ? body : body ? JSON.stringify(body) : undefined });
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const msg = (data && data.detail) ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : `Request failed (${res.status})`;
    throw new ApiError(res.status, msg);
  }
  return data;
}

export const api = {
  health: () => request("GET", "/health"),
  catalog: () => request("GET", "/catalog"),
  glossary: () => request("GET", "/glossary"),
  lessons: () => request("GET", "/lessons"),
  samples: () => request("GET", "/samples"),
  datasets: () => request("GET", "/datasets"),
  dataset: (id) => request("GET", `/datasets/${id}`),
  deleteDataset: (id) => request("DELETE", `/datasets/${id}`),
  upload: (file) => { const f = new FormData(); f.append("file", file); return request("POST", "/datasets/upload", f, true); },
  loadSample: (key) => request("POST", `/datasets/sample/${key}`),
  task: (id, target) => request("POST", `/datasets/${id}/task`, { target }),
  createJob: (kind, dataset_id, params) => request("POST", "/jobs", { kind, dataset_id, params }),
  job: (id) => request("GET", `/jobs/${id}`),
  jobs: (dataset_id) => request("GET", "/jobs" + (dataset_id ? `?dataset_id=${dataset_id}` : "")),
  models: (dataset_id) => request("GET", "/models" + (dataset_id ? `?dataset_id=${dataset_id}` : "")),
  predict: (id, rows) => request("POST", `/models/${id}/predict`, { rows }),
  predictFile: (id, file) => { const f = new FormData(); f.append("file", file); return request("POST", `/models/${id}/predict-file`, f, true); },
  modelCode: (id) => request("GET", `/models/${id}/code`),
};

// Poll a job until it finishes. onProgress gets the job row on every tick.
export async function waitForJob(id, onProgress, signal) {
  let delay = 700;
  for (;;) {
    if (signal && signal.aborted) throw new Error("cancelled");
    const j = await api.job(id);
    if (onProgress) onProgress(j);
    if (j.status === "done" || j.status === "error") return j;
    await new Promise((r) => setTimeout(r, delay));
    delay = Math.min(delay * 1.3, 3000);
  }
}

// Small cache for static reference data so views do not refetch.
const cache = {};
export async function cached(name) {
  if (!cache[name]) cache[name] = api[name]();
  try { return await cache[name]; } catch (e) { delete cache[name]; throw e; }
}
