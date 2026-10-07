// Predict a column: a linear flow. 1 pick column, 2 settings, 3 choose action, 4 result.
import { api, cached } from "../api.js";
import { el, clear, step, button, notice, table, select, field, details, codeBlock, metricTiles, narration, svgIcon, choices, fullCode, METRIC_NAMES, score, append } from "../ui.js";
import { barChart, lineChart, scatterChart, histogram, confusionMatrix } from "../charts.js";
import { runJob } from "../jobrun.js";

const state = {};

export async function render(root, [id], signal) {
  const [ds, catalog] = await Promise.all([api.dataset(id), cached("catalog")]);
  const S = state[id] = state[id] || { target: null, task: null, prep: { num_impute: "median", cat_impute: "most_frequent", scaler: "standard", encoder: "onehot", drop_columns: [] }, lastRace: null };
  const cols = ds.profile.columns;

  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("div", { class: "crumb" }, el("a", { href: "#/datasets", text: "Home" }), svgIcon("arrow", 12), el("a", { href: `#/dataset/${id}`, text: ds.name }), svgIcon("arrow", 12), el("span", { text: "Predict a column" })),
    el("h1", { text: "Predict a column" }))));

  const s2 = el("div"), s3 = el("div"), s4 = el("div");

  // ---- Step 1 --------------------------------------------------------------
  const taskBox = el("div");
  const targetSel = select([{ value: "", label: "Choose a column" }, ...cols.map((c) => ({ value: c.name, label: c.name }))], S.target || "", async (v) => {
    S.target = v || null; S.task = null; clear(taskBox); clear(s2); clear(s3); clear(s4);
    if (!v) return;
    const t = await api.task(id, v);
    S.task = t.task;
    const kindText = t.task === "classification" ? `"${v}" is a category, so the model will guess which group each row belongs to.` : `"${v}" is a number, so the model will guess a value for each row.`;
    append(taskBox, [notice("ok", kindText), t.alt ? el("div", { class: "row" }, el("span", { class: "muted small", text: "Wrong guess?" }), button(t.alt === "regression" ? "Treat it as a number" : "Treat it as a category", { kind: "small", onclick: () => { S.task = t.alt; buildStep2(); } })) : null]);
    buildStep2();
  });
  root.append(step(3, "Which column should the model predict?", [
    el("p", { class: "plain", text: "This is the answer column. The tool learns from the other columns to predict it." }),
    field("Column to predict", targetSel), taskBox]));
  root.append(s2, s3, s4);
  if (S.target) targetSel.dispatchEvent(new Event("change"));

  // ---- Step 2 --------------------------------------------------------------
  function buildStep2() {
    clear(s2); clear(s3); clear(s4);
    const opts = el("div", { class: "grid grid-3" },
      field("Blank numbers", select([{ value: "median", label: "fill with the middle value" }, { value: "mean", label: "fill with the average" }, { value: "most_frequent", label: "fill with the most common" }, { value: "constant", label: "fill with 0" }], S.prep.num_impute, (v) => (S.prep.num_impute = v))),
      field("Blank text", select([{ value: "most_frequent", label: "fill with the most common" }, { value: "constant", label: "fill with 'missing'" }], S.prep.cat_impute, (v) => (S.prep.cat_impute = v))),
      field("Scale numbers", select([{ value: "standard", label: "yes, standard (recommended)" }, { value: "minmax", label: "yes, 0 to 1" }, { value: "robust", label: "yes, ignore outliers" }, { value: "none", label: "no" }], S.prep.scaler, (v) => (S.prep.scaler = v)), "Puts all number columns on the same scale so none dominates."),
      field("Text columns", select([{ value: "onehot", label: "one column per value (recommended)" }, { value: "ordinal", label: "replace with numbers" }], S.prep.encoder, (v) => (S.prep.encoder = v))));
    const dropChips = el("div", { class: "chips" }, cols.filter((c) => c.name !== S.target).map((c) => chip(c.name, S.prep.drop_columns.includes(c.name), (on) => { S.prep.drop_columns = on ? [...new Set([...S.prep.drop_columns, c.name])] : S.prep.drop_columns.filter((x) => x !== c.name); })));
    s2.append(step(4, "Prepare the data", [
      el("p", { class: "plain", text: "Blanks get filled, numbers get scaled, and text gets turned into numbers. The default settings work for most files." }),
      details("Change settings", opts),
      details("Leave some columns out (for example IDs or names)", dropChips),
      button("Continue", { kind: "primary", icon: "arrow", onclick: buildStep3 })]));
  }

  // ---- Step 3 --------------------------------------------------------------
  function buildStep3() {
    clear(s3); clear(s4);
    const reg = catalog[S.task];
    const form = el("div");
    s3.append(step(5, "What do you want to do?", [
      choices([
        { key: "race", icon: "play", title: "Find the best model", text: "Try every model and rank them. Start here." },
        { key: "train", icon: "flask", title: "Train one model", text: "Pick a model and see full results and charts." },
        { key: "ensemble", icon: "layers", title: "Combine models", text: "Join several models into one stronger one." },
        { key: "tune", icon: "sliders", title: "Fine-tune a model", text: "Try different settings to squeeze out a better score." },
        { key: "curve", icon: "scatter", title: "Would more data help?", text: "See how the score changes with more rows." },
      ], (k) => { clear(form).append({ race: raceForm, train: trainForm, ensemble: ensembleForm, tune: tuneForm, curve: curveForm }[k](reg)); }),
      form]));
    s3.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function baseParams() { return { task: S.task, target: S.target, preprocess: S.prep }; }
  function showResult(node) { clear(s4).append(step(6, "Result", node)); s4.scrollIntoView({ behavior: "smooth", block: "start" }); }

  function raceForm(reg) {
    const chosen = new Set(reg.map((m) => m.key));
    const chips = el("div", { class: "chips" }, reg.map((m) => chip(m.name, true, (on) => (on ? chosen.add(m.key) : chosen.delete(m.key)), m.explain)));
    const slow = el("input", { type: "checkbox" });
    const host = el("div");
    const run = button("Run", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("leaderboard", id, { ...baseParams(), models: [...chosen], include_slow: slow.checked }, host, signal); S.lastRace = job.result; showResult(raceResult(job.result, reg)); }
      catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" },
      el("p", { class: "plain", text: "Every model gets the same test. You get a ranked list in about a minute." }),
      details("Choose which models to try", [chips, el("label", { class: "check" }, slow, "Also try the slow models on big files")]),
      run, host);
  }

  function trainForm(reg, preset) {
    const modelSel = select(reg.map((m) => ({ value: m.key, label: m.name })), preset?.model || S.lastRace?.leaderboard?.[0]?.key || reg[0].key, () => refreshParams());
    const explain = el("p", { class: "muted small" });
    const paramsBox = el("div", { class: "grid grid-3" });
    const paramVals = {};
    function refreshParams() {
      const m = reg.find((x) => x.key === modelSel.value);
      explain.textContent = m.explain;
      clear(paramsBox);
      for (const k in paramVals) delete paramVals[k];
      for (const p of m.tunable) paramsBox.append(field(p, el("input", { type: "text", placeholder: "default", oninput: (e) => (paramVals[p] = parse(e.target.value)) }), m.params[p] || ""));
    }
    refreshParams();
    const testSize = select([{ value: 0.2, label: "20% (recommended)" }, { value: 0.1, label: "10%" }, { value: 0.3, label: "30%" }, { value: 0.4, label: "40%" }], 0.2);
    const host = el("div");
    const run = button("Train", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try {
        const spec = { kind: "single", model: modelSel.value, params: Object.fromEntries(Object.entries(paramVals).filter(([, v]) => v !== undefined)) };
        const job = await runJob("train", id, { ...baseParams(), spec, test_size: Number(testSize.value) }, host, signal);
        showResult(trainResult(job.result));
      } catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" },
      el("div", { class: "grid grid-3" }, field("Model", modelSel), field("Rows kept aside for testing", testSize, "The model never sees these rows while learning.")),
      explain, details("Advanced settings (leave blank for defaults)", paramsBox), run, host);
  }

  function ensembleForm(reg) {
    const kindSel = select([{ value: "voting", label: "Vote: models vote on the answer" }, { value: "stacking", label: "Stack: one model learns whom to trust" }, { value: "bagging", label: "Bag: many copies of one model" }], "voting", () => refresh());
    const explain = el("p", { class: "muted small" });
    const body = el("div", { class: "stack" });
    const members = new Set(["logreg", "rf", "knn"].filter((k) => reg.some((m) => m.key === k)));
    let finalKey = S.task === "classification" ? "logreg" : "ridge", baseKey = "tree", nEst = 25, voting = "soft";
    function refresh() {
      explain.textContent = catalog.ensembles[kindSel.value];
      clear(body);
      if (kindSel.value === "bagging") {
        body.append(el("div", { class: "grid grid-3" }, field("Model to copy", select(reg.map((m) => ({ value: m.key, label: m.name })), baseKey, (v) => (baseKey = v))), field("How many copies", el("input", { type: "number", min: 2, max: 100, value: nEst, oninput: (e) => (nEst = Number(e.target.value)) }))));
      } else {
        body.append(el("div", { class: "field" }, el("span", { class: "field-label", text: "Pick the models to combine" }), el("div", { class: "chips" }, reg.map((m) => chip(m.name, members.has(m.key), (on) => (on ? members.add(m.key) : members.delete(m.key)), m.explain)))));
        if (kindSel.value === "stacking") body.append(field("Final model", select(reg.map((m) => ({ value: m.key, label: m.name })), finalKey, (v) => (finalKey = v))));
        else if (S.task === "classification") body.append(field("How to vote", select([{ value: "soft", label: "average the confidence (recommended)" }, { value: "hard", label: "majority wins" }], voting, (v) => (voting = v))));
      }
    }
    refresh();
    const host = el("div");
    const run = button("Train the combined model", { kind: "primary", icon: "layers", onclick: async () => {
      run.disabled = true;
      try {
        const spec = kindSel.value === "bagging" ? { kind: "bagging", model: baseKey, n_estimators: nEst } : { kind: kindSel.value, members: [...members], final: finalKey, voting };
        const job = await runJob("train", id, { ...baseParams(), spec }, host, signal);
        showResult(trainResult(job.result));
      } catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, field("How to combine", kindSel), explain, body, run, host);
  }

  function tuneForm(reg) {
    const tunable = reg.filter((m) => m.tunable.length);
    const modelSel = select(tunable.map((m) => ({ value: m.key, label: m.name })), S.lastRace?.leaderboard?.find((r) => r.status === "ok" && tunable.some((m) => m.key === r.key))?.key || tunable[0].key);
    const nIter = select([{ value: 8, label: "8 tries (fast)" }, { value: 12, label: "12 tries" }, { value: 20, label: "20 tries (slow)" }], 12);
    const host = el("div");
    const run = button("Start tuning", { kind: "primary", icon: "sliders", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("tune", id, { ...baseParams(), model: modelSel.value, n_iter: Number(nIter.value) }, host, signal); showResult(tuneResult(job.result)); }
      catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, el("p", { class: "plain", text: "The tool tries random settings for the model and keeps the best." }), el("div", { class: "grid grid-3" }, field("Model", modelSel), field("How many settings to try", nIter)), run, host);
  }

  function curveForm(reg) {
    const modelSel = select(reg.map((m) => ({ value: m.key, label: m.name })), S.lastRace?.leaderboard?.[0]?.key || reg[0].key);
    const host = el("div");
    const run = button("Check", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("curve", id, { ...baseParams(), model: modelSel.value }, host, signal); showResult(curveResult(job.result)); }
      catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, el("p", { class: "plain", text: "The model is trained on 15%, 36%, 57%, 79% and 100% of your rows. If the score is still rising at 100%, more data would help." }), field("Model", modelSel), run, host);
  }

  // ---- Results ---------------------------------------------------------------
  function raceResult(r, reg) {
    const ok = r.leaderboard.filter((x) => x.status === "ok");
    const items = r.leaderboard.map((x) => ({ label: x.name, value: x.status === "ok" ? x.mean : null, err: x.std, note: x.status === "skipped" ? "skipped" : x.status === "error" ? "failed" : "", color: famColor(x.family) }));
    const rows = r.leaderboard.map((x, i) => [
      el("span", { class: i === 0 && x.status === "ok" ? "winner" : "" }, x.name), famName(x.family),
      x.status === "ok" ? x.mean.toFixed(4) : el("span", { class: "muted", text: x.status === "skipped" ? "skipped" : "failed" }),
      x.status === "ok" ? `± ${x.std.toFixed(4)}` : "", x.status === "ok" ? `${x.seconds}s` : (x.note || ""),
      x.status === "ok" ? button("Train this", { kind: "small", onclick: () => { clear(s3); s3.append(step(5, `Train ${x.name}`, trainForm(reg, { model: x.key }))); s3.scrollIntoView({ behavior: "smooth" }); } }) : "",
    ]);
    return [
      ok.length ? el("p", { class: "big-answer", text: `Best model: ${ok[0].name} (${METRIC_NAMES[r.scoring] || r.scoring} ${score(ok[0].mean)})` }) : null,
      r.subsampled ? notice("info", `To stay fast, this used ${r.rows_used.toLocaleString()} of your ${r.rows_total.toLocaleString()} rows. Train the winner to use all rows.`) : null,
      el("div", { class: "stack" }, r.summary.map((t) => el("p", { text: t }))),
      el("div", { class: "chart-wrap" }, barChart(items, { title: "Ranking" })),
      el("p", { class: "chart-note", text: "Longer bar is better. The small line shows how much the score moved between test rounds." }),
      details("Full table", table(["Model", "Type", "Score", "Varies by", "Time", ""], rows)),
      details("How this was done, step by step", narration(r.narration), true),
      details("How the data was prepared", prepSteps(r.preprocessing)),
      fullCode(r),
    ];
  }

  function trainResult(r) {
    const ev = r.evaluation;
    const diag = ev.fit_diagnosis || {};
    const diagClass = diag.label === "good fit" ? "diag-good" : diag.label?.includes("overfit") ? "diag-mid" : "diag-bad";
    const verdict = { "good fit": "Good: the model works about as well on new rows as on the ones it learned from.", overfitting: "Careful: the model memorised its training rows and does worse on new ones.", "slight overfitting": "Mostly fine: a small drop on new rows.", underfitting: "Weak: the model is too simple for this data." }[diag.label] || diag.text;
    const parts = [
      el("p", { class: "big-answer", text: `${r.model_name}: ${METRIC_NAMES[ev.primary] || ev.primary} ${score(ev.metrics[ev.primary])} on rows it never saw` }),
      el("p", {}, el("strong", { class: diagClass, text: `${verdict} ` }), el("span", { class: "muted small", text: `(training rows ${score(ev.train_score)}, test rows ${score(ev.metrics[ev.primary])})` })),
      metricTiles(ev.metrics, ev.how_to_read, ev.primary),
      el("p", { class: "muted small", text: "Hover a box to see what the number means." }),
    ];
    const charts = [];
    if (r.task === "classification") {
      charts.push(chartCard("Right and wrong guesses", confusionMatrix(ev.confusion, ev.classes), "Rows are the true answer, columns are the model's guess. Green diagonal is correct."));
      if (ev.roc) charts.push(chartCard("ROC curve", lineChart([{ name: "model", points: ev.roc }], { xd: [0, 1], yd: [0, 1], diagonal: true, xlabel: "false alarms", ylabel: "caught", height: 260 }), "Closer to the top-left corner is better. The dotted line is random guessing."));
      parts.push(details("Score for each category", table(["Category", { label: "Precision", key: "precision", num: true }, { label: "Recall", key: "recall", num: true }, { label: "F1", key: "f1", num: true }, { label: "Rows", key: "support", num: true }], ev.per_class.map((c) => ({ ...c, Category: c.class })))));
    } else {
      charts.push(chartCard("Guess vs real value", scatterChart(ev.pred_vs_actual, null, { diagonal: true, xlabel: "real", ylabel: "guess", height: 300 }), "Each dot is one test row. Dots on the dotted line are perfect guesses."));
      charts.push(chartCard("Size of the errors", histogram(ev.residual_hist, { zeroLine: true, xlabel: "real minus guess", height: 200 }), "Should be centred on zero. A lean to one side means the model is biased."));
    }
    if (ev.importance) charts.push(chartCard("Which columns matter most", barChart(ev.importance.items.map((i) => ({ label: i.feature, value: i.value })), { digits: 4 }), ev.importance.note));
    parts.push(el("div", { class: "grid grid-2" }, charts));
    if (r.model_id) parts.push(el("div", { class: "row" },
      el("a", { class: "btn primary", href: `#/dataset/${id}/predict` }, svgIcon("target"), el("span", { text: "Use this model on new rows" })),
      el("a", { class: "btn", href: `/api/models/${r.model_id}/download`, download: "" }, svgIcon("download"), el("span", { text: "Download model" })),
      el("a", { class: "btn", href: `/api/models/${r.model_id}/notebook`, download: "" }, svgIcon("book"), el("span", { text: "Download notebook" }))));
    parts.push(details("How this was done, step by step", narration(r.narration), true), details("How the data was prepared", prepSteps(r.preprocessing)), fullCode(r));
    return parts;
  }

  function tuneResult(r) {
    const best = r.trials[0];
    const gain = (best.mean || 0) - r.baseline;
    const items = r.trials.filter((t) => t.mean !== null).map((t) => ({ label: Object.entries(t.params).map(([k, v]) => `${k}=${fmtv(v)}`).join(", "), value: t.mean, err: t.std }));
    items.push({ label: "default settings", value: r.baseline, color: "#9ca3af" });
    return [
      el("p", { class: "big-answer", text: gain > 0.005 ? `Tuning improved ${r.model_name} from ${score(r.baseline)} to ${score(best.mean)}` : `Tuning did not help much: ${score(r.baseline)} to ${score(best.mean)}. The defaults were already good.` }),
      el("div", { class: "chart-wrap" }, barChart(items, { digits: 4 })),
      el("p", { class: "chart-note", text: "Each bar is one set of settings, best first. Grey is the untouched default." }),
      el("h4", { text: "Best settings" }), codeBlock(`best_params = ${JSON.stringify(best.params, null, 2)}`),
      Object.keys(r.param_help).length ? details("What each setting means", el("dl", { class: "glossary" }, Object.entries(r.param_help).flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]))) : null,
      details("How this was done, step by step", narration(r.narration)), fullCode(r),
    ];
  }

  function curveResult(r) {
    const pts = r.points;
    return [
      el("p", { class: "big-answer", text: r.verdict }),
      el("div", { class: "chart-wrap" }, lineChart([
        { name: "training rows", points: pts.map((p) => [p.n, p.train]) },
        { name: "new rows", points: pts.map((p) => [p.n, p.test]), band: pts.map((p) => [p.n, p.test - p.test_std, p.test + p.test_std]) },
      ], { xlabel: "rows used", ylabel: METRIC_NAMES[r.scoring] || r.scoring, height: 300, xfmt: (v) => Math.round(v) })),
      el("p", { class: "chart-note", text: "If the lower line is still climbing at the right edge, more data would help. If the two lines stay far apart, the model is memorising." }),
      details("How this was done, step by step", narration(r.narration)), fullCode(r),
    ];
  }
}

function chartCard(title, chart, note) { return el("div", { class: "col-card" }, el("h4", { text: title }), chart, el("p", { class: "chart-note", text: note })); }

export function prepSteps(prep) {
  if (!prep) return null;
  return el("div", { class: "stack" }, prep.steps.map((s) => el("div", {}, el("strong", { text: s.title }), el("p", { class: "small", text: s.why }), s.columns?.length ? el("p", { class: "small muted", text: "Columns: " + s.columns.join(", ") }) : null, codeBlock(s.code))));
}

export function chip(label, on, onchange, title) {
  const input = el("input", { type: "checkbox" }); input.checked = on;
  const c = el("label", { class: "chip " + (on ? "on" : ""), title: title || "" }, input, label);
  input.addEventListener("change", () => { c.classList.toggle("on", input.checked); onchange(input.checked); });
  return c;
}

function famColor(f) { return { linear: "#2563eb", distance: "#0891b2", probabilistic: "#7c3aed", tree: "#4d7c0f", kernel: "#be185d", neural: "#b45309", bagging: "#059669", boosting: "#d97706" }[f] || "#64748b"; }
function famName(f) { return { linear: "linear", distance: "nearest rows", probabilistic: "probability", tree: "decision tree", kernel: "support vector", neural: "neural net", bagging: "many trees (bagging)", boosting: "many trees (boosting)" }[f] || f; }

function parse(v) {
  v = v.trim();
  if (v === "") return undefined;
  if (v === "None" || v === "null") return null;
  if (v === "true") return true; if (v === "false") return false;
  if (/^-?\d+$/.test(v)) return parseInt(v, 10);
  if (/^-?\d*\.\d+(e-?\d+)?$/i.test(v)) return parseFloat(v);
  if (/^\(.*\)$/.test(v)) return v.slice(1, -1).split(",").map((x) => parseInt(x, 10)).filter((x) => !isNaN(x));
  return v;
}
function fmtv(v) { return typeof v === "number" && !Number.isInteger(v) ? v.toPrecision(3) : Array.isArray(v) ? `(${v.join(",")})` : String(v); }
