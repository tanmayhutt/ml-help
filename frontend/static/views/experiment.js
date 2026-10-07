// Supervised lab: target -> task -> preprocessing -> race / train / ensemble / tune / learning curve.
import { api, cached } from "../api.js";
import { el, clear, card, button, notice, table, select, field, details, codeBlock, metricTiles, narration, tabs, svgIcon, toast, fullCode } from "../ui.js";
import { barChart, lineChart, scatterChart, histogram, confusionMatrix } from "../charts.js";
import { runJob } from "../jobrun.js";

const state = {};

export async function render(root, [id], signal) {
  const [ds, catalog] = await Promise.all([api.dataset(id), cached("catalog")]);
  const S = state[id] = state[id] || { target: null, task: null, prep: { num_impute: "median", cat_impute: "most_frequent", scaler: "standard", encoder: "onehot", drop_columns: [] }, metric: null, lastRace: null };
  const cols = ds.profile.columns;

  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("div", { class: "crumb" }, el("a", { href: "#/datasets", text: "Datasets" }), svgIcon("arrow", 12), el("a", { href: `#/dataset/${id}`, text: ds.name }), svgIcon("arrow", 12), el("span", { text: "Supervised" })),
    el("h1", { text: "Supervised learning" }),
    el("p", { text: "Choose what to predict, decide how the data is prepared, then race the whole model shelf. Every step explains itself." }))));

  // Step 1: target
  const taskBox = el("div");
  const targetSel = select([{ value: "", label: "Choose a column" }, ...cols.map((c) => ({ value: c.name, label: `${c.name} (${c.type}, ${c.unique} distinct)` }))], S.target || "", async (v) => {
    S.target = v || null; S.task = null; clear(taskBox);
    if (!v) return;
    const t = await api.task(id, v);
    S.task = t.task;
    const alt = t.alt ? el("div", { class: "row" }, el("span", { class: "muted small", text: "Not right?" }), button(`Treat as ${t.alt}`, { kind: "small", onclick: () => { S.task = t.alt; clear(taskBox).append(notice("info", `Treating '${v}' as ${t.alt}.`)); refreshStep3(); } })) : null;
    taskBox.append(notice(t.warning ? "warn" : "ok", [el("strong", { text: t.task + ". " }), t.reason]), alt);
    refreshStep3();
  });
  root.append(card("1. What should the model predict?", [
    el("p", { class: "muted small", text: "The target is the answer column. Everything else becomes a feature. If the target is a category you get classification; if it is a quantity you get regression." }),
    field("Target column", targetSel), taskBox]));
  if (S.target) targetSel.dispatchEvent(new Event("change"));

  // Step 2: preprocessing
  const prepBody = el("div", { class: "grid grid-3" },
    field("Missing numbers", select(["median", "mean", "most_frequent", "constant"], S.prep.num_impute, (v) => (S.prep.num_impute = v)), "Median resists outliers. Constant fills with 0."),
    field("Missing categories", select(["most_frequent", "constant"], S.prep.cat_impute, (v) => (S.prep.cat_impute = v)), "Constant fills with the word 'missing'."),
    field("Scale numbers", select(["standard", "minmax", "robust", "none"], S.prep.scaler, (v) => (S.prep.scaler = v)), "Essential for KNN, SVM, logistic, MLP. Irrelevant for trees."),
    field("Encode text", select(["onehot", "ordinal"], S.prep.encoder, (v) => (S.prep.encoder = v)), "One-hot avoids inventing an order."),
  );
  const dropChips = el("div", { class: "chips" }, cols.map((c) => chip(c.name, S.prep.drop_columns.includes(c.name), (on) => { S.prep.drop_columns = on ? [...new Set([...S.prep.drop_columns, c.name])] : S.prep.drop_columns.filter((x) => x !== c.name); })));
  root.append(card("2. How should the data be prepared?", [prepBody, details("Drop columns (IDs, leaked answers, free text)", dropChips)]));

  // Step 3: actions
  const step3 = el("div");
  root.append(step3);
  const results = el("div");
  root.append(results);

  function refreshStep3() {
    clear(step3);
    if (!S.task) { step3.append(card("3. Run experiments", notice("info", "Pick a target first."))); return; }
    const reg = catalog[S.task];
    const metricOpts = S.task === "classification" ? ["accuracy", "f1_macro", "balanced_accuracy"] : ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"];
    S.metric = S.metric && metricOpts.includes(S.metric) ? S.metric : metricOpts[0];
    const metricSel = field("Score models by", select(metricOpts, S.metric, (v) => (S.metric = v)), S.task === "classification" ? "Use f1_macro or balanced accuracy when classes are imbalanced." : "r2 is scale-free; MAE and RMSE are in target units (shown negated so higher is better).");
    const items = [
      { label: "Race all models", render: () => raceTab(reg, metricSel) },
      { label: "Train one model", render: () => trainTab(reg) },
      { label: "Build an ensemble", render: () => ensembleTab(reg) },
      { label: "Tune hyperparameters", render: () => tuneTab(reg) },
      { label: "Learning curve", render: () => curveTab(reg) },
    ];
    step3.append(card("3. Run experiments", tabs(items)));
  }

  function baseParams() { return { task: S.task, target: S.target, preprocess: S.prep, metric: S.metric }; }

  function raceTab(reg, metricSel) {
    const chosen = new Set(reg.map((m) => m.key));
    const chips = el("div", { class: "chips" }, reg.map((m) => chip(`${m.name}${m.slow ? " (slow)" : ""}`, true, (on) => (on ? chosen.add(m.key) : chosen.delete(m.key)), m.explain)));
    const slow = el("input", { type: "checkbox" });
    const host = el("div");
    const run = button("Race the models", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try {
        const job = await runJob("leaderboard", id, { ...baseParams(), models: [...chosen], include_slow: slow.checked }, host, signal);
        S.lastRace = job.result;
        clear(results).append(raceResult(job.result, reg));
        results.scrollIntoView({ behavior: "smooth" });
      } catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" },
      el("p", { class: "muted", text: `Every selected model is cross-validated on the same ${S.task} problem and ranked. Big datasets are subsampled for the race so it finishes in seconds; train the winner on full data afterwards.` }),
      metricSel, chips,
      el("label", { class: "check" }, slow, "Include slow models (SVM, KNN, MLP) even on large data"),
      run, host);
  }

  function trainTab(reg, preset) {
    const modelSel = select(reg.map((m) => ({ value: m.key, label: m.name })), preset?.model || S.lastRace?.leaderboard?.[0]?.key || reg[0].key, () => refreshParams());
    const explain = el("p", { class: "muted small" });
    const paramsBox = el("div", { class: "grid grid-3" });
    const paramVals = {};
    function refreshParams() {
      const m = reg.find((x) => x.key === modelSel.value);
      explain.textContent = m.explain;
      clear(paramsBox);
      for (const k in paramVals) delete paramVals[k];
      for (const p of m.tunable) {
        const inp = el("input", { type: "text", placeholder: "default", oninput: (e) => (paramVals[p] = parse(e.target.value)) });
        paramsBox.append(field(p, inp, m.params[p] || ""));
      }
    }
    refreshParams();
    const testSize = select([0.1, 0.2, 0.3, 0.4], 0.2);
    const host = el("div");
    const run = button("Train and evaluate", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try {
        const spec = { kind: "single", model: modelSel.value, params: Object.fromEntries(Object.entries(paramVals).filter(([, v]) => v !== undefined)) };
        const job = await runJob("train", id, { ...baseParams(), spec, test_size: Number(testSize.value) }, host, signal);
        clear(results).append(trainResult(job.result, job.id));
        results.scrollIntoView({ behavior: "smooth" });
      } catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" },
      el("p", { class: "muted", text: "Hold out a test slice, fit one model on the rest, and read every metric and chart with a how-to-read note. Leave parameters blank for defaults." }),
      el("div", { class: "grid grid-3" }, field("Model", modelSel), field("Test share", testSize, "Rows kept hidden from training.")),
      explain, paramsBox, run, host);
  }

  function ensembleTab(reg) {
    const kindSel = select([{ value: "voting", label: "Voting" }, { value: "stacking", label: "Stacking" }, { value: "bagging", label: "Bagging" }], "voting", () => refresh());
    const explain = el("p", { class: "muted small" });
    const body = el("div", { class: "stack" });
    const members = new Set(["logreg", "rf", "knn"].filter((k) => reg.some((m) => m.key === k)));
    let finalKey = S.task === "classification" ? "logreg" : "ridge", baseKey = "tree", nEst = 25, voting = "soft";
    function refresh() {
      explain.textContent = catalog.ensembles[kindSel.value];
      clear(body);
      if (kindSel.value === "bagging") {
        body.append(el("div", { class: "grid grid-3" },
          field("Base model", select(reg.map((m) => ({ value: m.key, label: m.name })), baseKey, (v) => (baseKey = v)), "Decision tree is the classic; bagging a tree is almost a Random Forest."),
          field("Copies", el("input", { type: "number", min: 2, max: 100, value: nEst, oninput: (e) => (nEst = Number(e.target.value)) }), "Each trained on a bootstrap resample.")));
      } else {
        body.append(el("div", { class: "field" }, el("span", { class: "field-label", text: "Members" }),
          el("div", { class: "chips" }, reg.map((m) => chip(m.name, members.has(m.key), (on) => (on ? members.add(m.key) : members.delete(m.key)), m.explain)))));
        if (kindSel.value === "stacking") body.append(field("Meta-learner", select(reg.map((m) => ({ value: m.key, label: m.name })), finalKey, (v) => (finalKey = v)), "Learns how much to trust each member. Keep it simple."));
        else if (S.task === "classification") body.append(field("Voting", select(["soft", "hard"], voting, (v) => (voting = v)), "Soft averages probabilities and usually wins. Hard counts votes."));
      }
    }
    refresh();
    const host = el("div");
    const run = button("Train the ensemble", { kind: "primary", icon: "layers", onclick: async () => {
      run.disabled = true;
      try {
        const spec = kindSel.value === "bagging" ? { kind: "bagging", model: baseKey, n_estimators: nEst } : { kind: kindSel.value, members: [...members], final: finalKey, voting };
        const job = await runJob("train", id, { ...baseParams(), spec }, host, signal);
        clear(results).append(trainResult(job.result, job.id));
        results.scrollIntoView({ behavior: "smooth" });
      } catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, field("Ensemble type", kindSel), explain, body, run, host);
  }

  function tuneTab(reg) {
    const tunable = reg.filter((m) => m.tunable.length);
    const modelSel = select(tunable.map((m) => ({ value: m.key, label: m.name })), S.lastRace?.leaderboard?.find((r) => r.status === "ok" && tunable.some((m) => m.key === r.key))?.key || tunable[0].key, () => (explain.textContent = describe()));
    const nIter = el("input", { type: "number", min: 2, max: 20, value: 12 });
    const explain = el("p", { class: "muted small" });
    function describe() { const m = tunable.find((x) => x.key === modelSel.value); return `Searches: ${m.tunable.join(", ")}. ` + Object.entries(m.params).map(([k, v]) => `${k}: ${v}`).join(" "); }
    explain.textContent = describe();
    const host = el("div");
    const run = button("Run random search", { kind: "primary", icon: "sliders", onclick: async () => {
      run.disabled = true;
      try {
        const job = await runJob("tune", id, { ...baseParams(), model: modelSel.value, n_iter: Number(nIter.value) }, host, signal);
        clear(results).append(tuneResult(job.result));
        results.scrollIntoView({ behavior: "smooth" });
      } catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" },
      el("p", { class: "muted", text: "Random search samples configurations from each model's parameter space and cross-validates each one. Capped at 20 trials so a search never takes more than a couple of minutes." }),
      el("div", { class: "grid grid-3" }, field("Model", modelSel), field("Trials", nIter, "Each trial costs 3 fits.")), explain, run, host);
  }

  function curveTab(reg) {
    const modelSel = select(reg.map((m) => ({ value: m.key, label: m.name })), S.lastRace?.leaderboard?.[0]?.key || reg[0].key);
    const host = el("div");
    const run = button("Draw learning curve", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try {
        const job = await runJob("curve", id, { ...baseParams(), model: modelSel.value }, host, signal);
        clear(results).append(curveResult(job.result));
        results.scrollIntoView({ behavior: "smooth" });
      } catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" },
      el("p", { class: "muted", text: "Retrains on 15%, 36%, 57%, 79% and 100% of the training data and plots train versus test score. Tells you whether more data would help." }),
      field("Model", modelSel), run, host);
  }

  // ---- result renderers -------------------------------------------------
  function raceResult(r, reg) {
    const ok = r.leaderboard.filter((x) => x.status === "ok");
    const items = r.leaderboard.map((x, i) => ({ label: x.name, value: x.status === "ok" ? x.mean : null, err: x.std, note: x.status === "skipped" ? "skipped" : x.status === "error" ? "error" : "", color: famColor(x.family), dim: i > 0 && x.status === "ok" && false }));
    const rows = r.leaderboard.map((x, i) => [
      el("span", { class: i === 0 && x.status === "ok" ? "winner" : "" }, x.name), x.family,
      x.status === "ok" ? x.mean.toFixed(4) : el("span", { class: "muted", text: x.status }),
      x.status === "ok" ? `± ${x.std.toFixed(4)}` : "", x.status === "ok" ? `${x.seconds}s` : (x.note || ""),
      x.status === "ok" ? button("Train", { kind: "small", onclick: () => { clear(results).append(card(`Train ${x.name}`, trainTab(reg, { model: x.key }))); } }) : "",
    ]);
    return card(`Leaderboard: ${r.scoring} across ${r.folds} folds`, [
      r.subsampled ? notice("info", `Raced on a ${r.rows_used.toLocaleString()}-row sample of ${r.rows_total.toLocaleString()} rows to stay fast.`) : null,
      el("div", { class: "stack" }, r.summary.map((t) => el("p", { text: t }))),
      el("div", { class: "chart-wrap" }, barChart(items, { title: "Leaderboard" })),
      el("p", { class: "chart-note", text: "Bars are mean CV score, whiskers are one standard deviation across folds. Colour is the model family." }),
      table(["Model", "Family", "Score", "Spread", "Time", ""], rows),
      details("Step by step", narration(r.narration), true),
      details("Preprocessing that was applied", prepSteps(r.preprocessing)),
      fullCode(r),
    ], { aside: el("span", { class: "badge", text: `${ok.length} finished` }) });
  }

  function trainResult(r, jobId) {
    const ev = r.evaluation;
    const diag = ev.fit_diagnosis || {};
    const diagClass = diag.label === "good fit" ? "diag-good" : diag.label?.includes("overfit") ? "diag-mid" : "diag-bad";
    const parts = [
      el("div", { class: "row" }, el("span", { class: "badge accent", text: r.task }), el("span", { class: "muted small", text: `${r.rows_used.toLocaleString()} rows, ${Math.round(r.test_size * 100)}% held out` })),
      metricTiles(ev.metrics, ev.how_to_read, ev.primary),
      el("p", {}, el("strong", { class: diagClass, text: `${diag.label}. ` }), `Train ${ev.primary} ${ev.train_score}, test ${ev.metrics[ev.primary]}. ${diag.text}`),
    ];
    const charts = [];
    if (r.task === "classification") {
      charts.push(chartCard("Confusion matrix", confusionMatrix(ev.confusion, ev.classes), ev.how_to_read.confusion));
      if (ev.roc) charts.push(chartCard("ROC curve", lineChart([{ name: "model", points: ev.roc }], { xd: [0, 1], yd: [0, 1], diagonal: true, xlabel: "false positive rate", ylabel: "true positive rate", height: 260 }), ev.how_to_read.roc + (ev.roc_note ? " " + ev.roc_note : "")));
      if (ev.pr) charts.push(chartCard("Precision-recall", lineChart([{ name: "model", points: ev.pr }], { xd: [0, 1], yd: [0, 1], xlabel: "recall", ylabel: "precision", height: 260 }), ev.how_to_read.pr));
      parts.push(details("Per-class report", table(["Class", { label: "Precision", key: "precision", num: true }, { label: "Recall", key: "recall", num: true }, { label: "F1", key: "f1", num: true }, { label: "Rows", key: "support", num: true }], ev.per_class.map((c) => ({ ...c, Class: c.class })))));
    } else {
      charts.push(chartCard("Predicted vs actual", scatterChart(ev.pred_vs_actual, null, { diagonal: true, xlabel: "actual", ylabel: "predicted", height: 300 }), ev.how_to_read.pred_vs_actual));
      charts.push(chartCard("Residuals", histogram(ev.residual_hist, { zeroLine: true, xlabel: "actual minus predicted", height: 200 }), ev.how_to_read.residual_hist));
    }
    if (ev.importance) charts.push(chartCard(`Feature importance (${ev.importance.kind})`, barChart(ev.importance.items.map((i) => ({ label: i.feature, value: i.value })), { digits: 4 }), ev.importance.note));
    parts.push(el("div", { class: "grid grid-2" }, charts));
    const actions = el("div", { class: "row" });
    if (r.model_id) {
      actions.append(
        el("a", { class: "btn", href: `/api/models/${r.model_id}/download`, download: "" }, svgIcon("download"), el("span", { text: "Download model (.joblib)" })),
        el("a", { class: "btn", href: `/api/models/${r.model_id}/notebook`, download: "" }, svgIcon("book"), el("span", { text: "Download notebook (.ipynb)" })),
        el("a", { class: "btn", href: `#/dataset/${id}/predict` }, svgIcon("target"), el("span", { text: "Predict with this model" })),
      );
    }
    parts.push(actions);
    parts.push(details("Step by step", narration(r.narration), true));
    parts.push(details("Preprocessing that was applied", prepSteps(r.preprocessing)));
    parts.push(fullCode(r));
    return card(r.model_name, parts);
  }

  function tuneResult(r) {
    const best = r.trials[0];
    const items = r.trials.filter((t) => t.mean !== null).map((t, i) => ({ label: Object.entries(t.params).map(([k, v]) => `${k}=${fmtv(v)}`).join(", "), value: t.mean, err: t.std }));
    items.push({ label: "defaults", value: r.baseline, color: "#9ca3af" });
    return card(`Tuning ${r.model_name}`, [
      el("div", { class: "stack" }, r.narration.map((n) => el("p", { text: n.text }))),
      el("div", { class: "chart-wrap" }, barChart(items, { digits: 4 })),
      el("p", { class: "chart-note", text: `${r.scoring} per trial, best first. The grey bar is the untuned default.` }),
      codeBlock(`best_params = ${JSON.stringify(best.params, null, 2)}`),
      Object.keys(r.param_help).length ? details("What the parameters mean", el("dl", { class: "glossary" }, Object.entries(r.param_help).flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]))) : null,
      details("Step by step", narration(r.narration)),
      fullCode(r),
    ]);
  }

  function curveResult(r) {
    const pts = r.points;
    return card(`Learning curve: ${r.model_name}`, [
      el("p", {}, el("strong", { text: "Verdict. " }), r.verdict),
      el("div", { class: "chart-wrap" }, lineChart([
        { name: "train", points: pts.map((p) => [p.n, p.train]) },
        { name: "test (cv)", points: pts.map((p) => [p.n, p.test]), band: pts.map((p) => [p.n, p.test - p.test_std, p.test + p.test_std]) },
      ], { xlabel: "training rows", ylabel: r.scoring, height: 300, xfmt: (v) => Math.round(v) })),
      el("p", { class: "chart-note", text: "A wide gap that stays wide is variance (overfitting). Two low curves close together is bias (underfitting). The band is one standard deviation across folds." }),
      details("Step by step", narration(r.narration)),
      fullCode(r),
    ]);
  }
}

function chartCard(title, chart, note) {
  return el("div", { class: "col-card" }, el("h4", { text: title }), chart, el("p", { class: "chart-note", text: note }));
}

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

function famColor(f) {
  return { linear: "#2563eb", distance: "#0891b2", probabilistic: "#7c3aed", tree: "#4d7c0f", kernel: "#be185d", neural: "#b45309", bagging: "#059669", boosting: "#d97706" }[f] || "#64748b";
}

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
