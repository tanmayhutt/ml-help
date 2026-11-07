// Predict a column: a linear flow. 1 pick column, 2 settings, 3 choose action, 4 result.
import { api, cached } from "../api.js";
import { el, clear, step, button, notice, table, select, field, details, codeBlock, metricTiles, narration, svgIcon, choices, fullCode, METRIC_NAMES, score, append } from "../ui.js";
import { barChart, lineChart, scatterChart, histogram, confusionMatrix } from "../charts.js";
import { runJob } from "../jobrun.js";
import { tryModelPanel } from "../trymodel.js";

const state = {};

export async function render(root, [id], signal) {
  const [ds, catalog] = await Promise.all([api.dataset(id), cached("catalog")]);
  let planOpts = null;
  try { const raw = sessionStorage.getItem(`mlhelp.plan.${id}`); if (raw) { planOpts = JSON.parse(raw); sessionStorage.removeItem(`mlhelp.plan.${id}`); } } catch {}
  const recommended = planOpts || ds.profile.cleaning?.options || {};
  const S = state[id] = state[id] || { target: null, task: null, prep: { num_impute: "median", cat_impute: "most_frequent", scaler: "standard", encoder: "onehot", drop_columns: [], drop_duplicates: false, outliers: "none", log_skewed: false, expand_dates: true, rare_min: 5, ordered_auto: true, drop_correlated: 0.95, balance: "auto", log_target: "auto", ...recommended }, refine: { feature_select: true, tune_winner: true }, lastRace: null };
  if (planOpts) { S.prep = { ...S.prep, ...planOpts }; S.planApplied = true; }
  const cols = ds.profile.columns;

  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("div", { class: "crumb" }, el("a", { href: "#/datasets", text: "Home" }), svgIcon("arrow", 12), el("a", { href: `#/dataset/${id}`, text: ds.name }), svgIcon("arrow", 12), el("span", { text: "Predict a column" })),
    el("h1", { text: "Predict a column" }))));

  const s2 = el("div"), s3 = el("div"), s4 = el("div");

  // ---- Step 3: pick the column -------------------------------------------
  const taskBox = el("div");
  const suggestions = (ds.profile.suggested_targets || []).filter((t) => t.score >= 2).slice(0, 3);
  const pickTarget = async (v) => {
    S.target = v || null; S.task = null; clear(taskBox); clear(s2); clear(s3); clear(s4);
    [...root.querySelectorAll(".choice")].forEach((c) => c.classList.toggle("on", c.dataset.col === v));
    if (!v) return;
    const t = await api.task(id, v);
    S.task = t.task;
    const kindText = t.task === "classification" ? `"${v}" is a category, so the model will guess which group each row belongs to.` : `"${v}" is a number, so the model will guess a value for each row.`;
    append(taskBox, [notice("ok", kindText), t.alt ? el("div", { class: "row" }, el("span", { class: "muted small", text: "Wrong guess?" }), button(t.alt === "regression" ? "Treat it as a number" : "Treat it as a category", { kind: "small", onclick: () => { S.task = t.alt; buildStep2(); } })) : null]);
    buildStep2();
  };
  const suggestionButtons = suggestions.length ? el("div", { class: "choices" }, suggestions.map((sg, i) => el("button", { type: "button", class: "choice", dataset: { col: sg.column }, onclick: () => pickTarget(sg.column) },
    el("strong", {}, svgIcon("target", 18), sg.column), el("span", { text: (i === 0 ? "Most likely. " : "") + `A ${sg.kind}: ${sg.why}.` })))) : null;
  const targetSel = select([{ value: "", label: "Pick a different column" }, ...cols.map((c) => ({ value: c.name, label: c.name }))], "", (v) => pickTarget(v));
  root.append(step(3, "Which column should the model predict?", [
    el("p", { class: "plain", text: "This is the answer column. The tool learns from the other columns to predict it." }),
    suggestionButtons, field(suggestions.length ? "Or choose any column" : "Column to predict", targetSel), taskBox]));
  root.append(s2, s3, s4);
  if (S.target) pickTarget(S.target);

  // ---- Step 4: one button --------------------------------------------------
  function buildStep2() {
    clear(s2); clear(s3); clear(s4);
    const opts = el("div", { class: "grid grid-3" },
      field("Blank numbers", select([{ value: "median", label: "fill with the middle value" }, { value: "mean", label: "fill with the average" }, { value: "most_frequent", label: "fill with the most common" }, { value: "constant", label: "fill with 0" }], S.prep.num_impute, (v) => (S.prep.num_impute = v))),
      field("Blank text", select([{ value: "most_frequent", label: "fill with the most common" }, { value: "constant", label: "fill with 'missing'" }], S.prep.cat_impute, (v) => (S.prep.cat_impute = v))),
      field("Scale numbers", select([{ value: "standard", label: "yes, standard (recommended)" }, { value: "minmax", label: "yes, 0 to 1" }, { value: "robust", label: "yes, ignore outliers" }, { value: "none", label: "no" }], S.prep.scaler, (v) => (S.prep.scaler = v)), "Puts all number columns on the same scale so none dominates."),
      field("Text columns", select([{ value: "onehot", label: "one column per value (recommended)" }, { value: "ordinal", label: "replace with numbers" }], S.prep.encoder, (v) => (S.prep.encoder = v))),
      field("Duplicate rows", select([{ value: "1", label: "remove them" }, { value: "0", label: "keep them" }], S.prep.drop_duplicates ? "1" : "0", (v) => (S.prep.drop_duplicates = v === "1"))),
      field("Outliers", select([{ value: "cap", label: "cap extreme values" }, { value: "none", label: "leave as is" }], S.prep.outliers, (v) => (S.prep.outliers = v)), "Pulls values beyond 1.5 times the middle range back to the edge."),
      field("Long-tailed columns", select([{ value: "1", label: "log-transform them" }, { value: "0", label: "leave as is" }], S.prep.log_skewed ? "1" : "0", (v) => (S.prep.log_skewed = v === "1")), "Squashes a long tail of big values."),
      field("Date columns", select([{ value: "1", label: "turn into year, month, weekday, days" }, { value: "0", label: "leave out" }], S.prep.expand_dates ? "1" : "0", (v) => (S.prep.expand_dates = v === "1"))),
      field("Rare categories", select([{ value: "5", label: "group values seen under 5 times" }, { value: "10", label: "group values seen under 10 times" }, { value: "0", label: "keep every value" }], String(S.prep.rare_min), (v) => (S.prep.rare_min = Number(v)))),
      field("Ordered words (low, medium, high)", select([{ value: "1", label: "encode in order" }, { value: "0", label: "treat as plain categories" }], S.prep.ordered_auto ? "1" : "0", (v) => (S.prep.ordered_auto = v === "1"))),
      field("Near-duplicate number columns", select([{ value: "0.95", label: "drop one of each pair (above 0.95)" }, { value: "0.99", label: "only above 0.99" }, { value: "0", label: "keep all" }], String(S.prep.drop_correlated), (v) => (S.prep.drop_correlated = Number(v)))),
      S.task === "classification" ? field("Rare class", select([{ value: "auto", label: "weigh it more when under 35% (recommended)" }, { value: "on", label: "always weigh it more" }, { value: "off", label: "never" }], S.prep.balance, (v) => (S.prep.balance = v)), "Stops models from scoring well by always guessing the common class.") : null,
      S.task === "regression" ? field("Long-tailed answer column", select([{ value: "auto", label: "predict log(value) when skewed (recommended)" }, { value: "on", label: "always" }, { value: "off", label: "never" }], S.prep.log_target, (v) => (S.prep.log_target = v))) : null,
      field("Curves and interactions", select([{ value: "0", label: "off (recommended)" }, { value: "2", label: "add squares and pairs (degree 2)" }, { value: "3", label: "add cubes too (degree 3)" }, { value: "2i", label: "pairs only, no squares" }], S.prep.poly_degree ? (S.prep.poly_interaction_only ? "2i" : String(S.prep.poly_degree)) : "0", (v) => { S.prep.poly_interaction_only = v === "2i"; S.prep.poly_degree = v === "0" ? 0 : v === "2i" ? 2 : Number(v); }), "Lets straight-line models bend. Makes many more columns."),
      field("Cut numbers into bins", select([{ value: "0", label: "off (recommended)" }, { value: "3", label: "3 ranges" }, { value: "5", label: "5 ranges" }, { value: "10", label: "10 ranges" }], String(S.prep.bin_numeric), (v) => (S.prep.bin_numeric = Number(v))), "Each number becomes which range it falls in. Replaces scaling."),
      S.task === "classification" ? field("Rare class rows", select([{ value: "none", label: "leave as is (use class weights)" }, { value: "oversample", label: "copy rare rows" }, { value: "undersample", label: "drop common rows" }, { value: "smote", label: "create new rare rows (SMOTE)" }], S.prep.resample, (v) => (S.prep.resample = v)), "Only training rows are resampled. Turns class weights off.") : null,
      field("Drop columns the winner ignores", select([{ value: "1", label: "test it, keep if the score holds" }, { value: "0", label: "skip" }], S.refine.feature_select ? "1" : "0", (v) => (S.refine.feature_select = v === "1"))),
      field("Tune the winner's settings", select([{ value: "1", label: "try 8 settings (recommended)" }, { value: "0", label: "skip" }], S.refine.tune_winner ? "1" : "0", (v) => (S.refine.tune_winner = v === "1"))));
    const dropChips = el("div", { class: "chips" }, cols.filter((c) => c.name !== S.target).map((c) => chip(c.name, S.prep.drop_columns.includes(c.name), (on) => { S.prep.drop_columns = on ? [...new Set([...S.prep.drop_columns, c.name])] : S.prep.drop_columns.filter((x) => x !== c.name); })));
    const host = el("div");
    const run = button("Find the best model", { kind: "primary big", icon: "play", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("auto", id, { ...baseParams(), ...S.refine }, host, signal); S.lastRace = job.result; S.lastModelId = job.result.model_id || S.lastModelId; showResult(autoResult(job.result)); }
      catch (e) { console.error(e); } finally { run.disabled = false; }
    } });
    const planNote = S.planApplied ? notice("ok", "The cleaning plan from the data page is applied: " + planSummary(S.prep) + ".") : (Object.keys(recommended).length ? notice("info", "Recommended cleaning is applied automatically: " + planSummary(S.prep) + ". Change it below if you want.") : null);
    s2.append(step(4, "Find the best model", [
      el("p", { class: "plain", text: "One click. The data is cleaned and encoded, every model on the shelf is tested, the best ones are combined, the winner is refined, and you get a clear answer with charts, a saved model, and the code." }),
      planNote, run, host,
      details("Change how the data is prepared and refined (optional)", [el("p", { class: "muted small", text: "Blanks get filled, numbers get scaled, and text gets turned into numbers. The defaults work for most files." }), opts]),
      details("Leave some columns out (for example IDs or names)", dropChips),
    ]));
    s2.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function nextSteps(reg, bestKey) {
    const form = el("div");
    return step("", "What next?", [
      choices([
        { key: "tune", icon: "sliders", title: "Fine-tune the winner", text: "Try different settings to squeeze out a bit more." },
        { key: "train", icon: "flask", title: "Train a different model", text: "Pick any model and see its full results." },
        { key: "ensemble", icon: "layers", title: "Combine models your way", text: "Choose which models to join together." },
        { key: "curve", icon: "scatter", title: "Would more data help?", text: "See how the score changes with more rows." },
        { key: "validation", icon: "sliders", title: "Watch one setting", text: "Move a setting from simple to complex and see underfitting and overfitting." },
        { key: "explain", icon: "info", title: "Explain the model", text: "Which columns it relies on, how each one moves the prediction, and why one row got its answer." },
        { key: "select", icon: "list", title: "Pick the best columns", text: "Keep only the columns that carry signal and check the score holds." },
        ...(S.task === "classification" ? [{ key: "calibration", icon: "target", title: "Check its confidence", text: "Does 80% confident mean right 80% of the time?" }] : []),
        { key: "race", icon: "play", title: "Re-run with chosen models", text: "Pick exactly which models to test." },
      ], (k) => { clear(form).append({ race: raceForm, train: trainForm, ensemble: ensembleForm, tune: tuneForm, curve: curveForm, validation: validationForm, explain: explainForm, select: selectForm, calibration: calibrationForm }[k](reg, { model: bestKey })); }),
      form]);
  }

  function baseParams() { return { task: S.task, target: S.target, preprocess: S.prep }; }
  function showResult(node, extra) { clear(s4).append(step(5, "Result", node)); if (extra) s4.append(extra); s4.scrollIntoView({ behavior: "smooth", block: "start" }); }

  function raceForm(reg) {
    const chosen = new Set(reg.map((m) => m.key));
    const chips = el("div", { class: "chips" }, reg.map((m) => chip(m.name, true, (on) => (on ? chosen.add(m.key) : chosen.delete(m.key)), m.explain)));
    const slow = el("input", { type: "checkbox" });
    const host = el("div");
    const run = button("Run", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("leaderboard", id, { ...baseParams(), models: [...chosen], include_slow: slow.checked }, host, signal); S.lastRace = job.result; showResult(raceResult(job.result, reg), nextSteps(reg, job.result.leaderboard[0]?.key)); }
      catch (e) { console.error(e); } finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" },
      el("p", { class: "plain", text: "Every model gets the same test. You get a ranked list in about a minute." }),
      details("Choose which models to try", [chips, el("label", { class: "check" }, slow, "Also try the slow models on big files")]),
      run, host);
  }

  function trainForm(reg, preset) {
    const modelSel = select(reg.map((m) => ({ value: m.key, label: m.name })), (preset?.model && reg.some((m) => m.key === preset.model) ? preset.model : null) || S.lastRace?.leaderboard?.find((r) => r.status === "ok" && reg.some((m) => m.key === r.key))?.key || reg[0].key, () => refreshParams());
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
        S.lastModelId = job.result.model_id || S.lastModelId;
        showResult(trainResult(job.result), nextSteps(reg, modelSel.value));
      } catch (e) { console.error(e); } finally { run.disabled = false; }
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
        S.lastModelId = job.result.model_id || S.lastModelId;
        showResult(trainResult(job.result), nextSteps(reg));
      } catch (e) { console.error(e); } finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, field("How to combine", kindSel), explain, body, run, host);
  }

  function tuneForm(reg, preset) {
    const tunable = reg.filter((m) => m.tunable.length);
    const modelSel = select(tunable.map((m) => ({ value: m.key, label: m.name })), (preset?.model && tunable.some((m) => m.key === preset.model) ? preset.model : null) || S.lastRace?.leaderboard?.find((r) => r.status === "ok" && tunable.some((m) => m.key === r.key))?.key || tunable[0].key);
    const nIter = select([{ value: 8, label: "8 tries (fast)" }, { value: 12, label: "12 tries" }, { value: 20, label: "20 tries (slow)" }], 12);
    const searchSel = select([{ value: "random", label: "random search (recommended)" }, { value: "grid", label: "grid search: every combination of a few values" }], "random");
    const host = el("div");
    const run = button("Start tuning", { kind: "primary", icon: "sliders", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("tune", id, { ...baseParams(), model: modelSel.value, n_iter: Number(nIter.value), search: searchSel.value }, host, signal); showResult(tuneResult(job.result), nextSteps(reg, modelSel.value)); }
      catch (e) { console.error(e); } finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, el("p", { class: "plain", text: "The tool tries different settings for the model and keeps the best. Random search samples settings from ranges; grid search tries every combination of a few fixed values." }), el("div", { class: "grid grid-3" }, field("Model", modelSel), field("How to search", searchSel), field("How many settings to try", nIter, "For a grid this sets the size of the grid.")), run, host);
  }

  function curveForm(reg, preset) {
    const modelSel = select(reg.map((m) => ({ value: m.key, label: m.name })), (preset?.model && reg.some((m) => m.key === preset.model) ? preset.model : null) || S.lastRace?.leaderboard?.find((r) => r.status === "ok" && reg.some((m) => m.key === r.key))?.key || reg[0].key);
    const host = el("div");
    const run = button("Check", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("curve", id, { ...baseParams(), model: modelSel.value }, host, signal); showResult(curveResult(job.result), nextSteps(reg, modelSel.value)); }
      catch (e) { console.error(e); } finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, el("p", { class: "plain", text: "The model is trained on 15%, 36%, 57%, 79% and 100% of your rows. If the score is still rising at 100%, more data would help." }), field("Model", modelSel), run, host);
  }

  // ---- New actions: validation curve, explain, select, calibration -----------
  function validationForm(reg, preset) {
    const tunable = reg.filter((m) => m.tunable.length);
    const modelSel = select(tunable.map((m) => ({ value: m.key, label: m.name })), (preset?.model && tunable.some((m) => m.key === preset.model) ? preset.model : null) || tunable[0].key, () => refreshParams());
    const paramSel = select([], "");
    const help = el("p", { class: "muted small" });
    function refreshParams() {
      const m = tunable.find((x) => x.key === modelSel.value);
      clear(paramSel); m.tunable.forEach((p) => paramSel.append(el("option", { value: p, text: p })));
      help.textContent = m.params[paramSel.value] || "";
    }
    paramSel.addEventListener("change", () => { const m = tunable.find((x) => x.key === modelSel.value); help.textContent = m.params[paramSel.value] || ""; });
    refreshParams();
    const host = el("div");
    const run = button("Draw the validation curve", { kind: "primary", icon: "sliders", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("validation", id, { ...baseParams(), model: modelSel.value, param: paramSel.value }, host, signal); showResult(validationResult(job.result), nextSteps(reg, modelSel.value)); }
      catch (e) { console.error(e); } finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, el("p", { class: "plain", text: "One setting is moved from simple to complex while everything else stays fixed. You see where the model underfits, where it overfits, and the sweet spot in between." }),
      el("div", { class: "grid grid-3" }, field("Model", modelSel), field("Setting to move", paramSel)), help, run, host);
  }

  function explainForm(reg) {
    const host = el("div");
    const chosen = new Set();
    const chips = el("div", { class: "chips" }, cols.filter((c) => c.name !== S.target).map((c) => chip(c.name, false, (on) => (on ? chosen.add(c.name) : chosen.delete(c.name)))));
    const rowIdx = el("input", { type: "number", min: 0, max: Math.max(0, ds.profile.rows - 1), value: 0 });
    const head = ds.profile.head;
    const preview = el("div", { class: "table-scroll small" }, table(head.columns, head.rows.slice(0, 8).map((r, i) => r), { class: "compact", onRow: (r) => { rowIdx.value = head.rows.indexOf(r); } }));
    const modelNote = S.lastModelId ? notice("info", "Explains the model you trained most recently on this page. Train one first if you want a different model explained.") : notice("warn", "Train a model first (Find the best model or Train one model), then come back here.");
    const run = button("Explain the model", { kind: "primary", icon: "info", disabled: !S.lastModelId, onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("explain", id, { model_id: S.lastModelId, features: [...chosen], row_index: Number(rowIdx.value) }, host, signal); showResult(explainResult(job.result), nextSteps(reg)); }
      catch (e) { console.error(e); } finally { run.disabled = !S.lastModelId; }
    } });
    return el("div", { class: "stack" }, el("p", { class: "plain", text: "Three views inside the model: which columns it relies on, how the prediction moves as one column changes, and why one particular row got its answer." }), modelNote,
      details("Which columns to look at (optional, up to 6; the most relied-on are used otherwise)", chips),
      el("div", { class: "grid grid-3" }, field("Row to explain", rowIdx, "Row number in your file, counting from 0. Click a row below to pick it.")), preview, run, host);
  }

  function selectForm(reg, preset) {
    const modelSel = select(reg.map((m) => ({ value: m.key, label: m.name })), (preset?.model && reg.some((m) => m.key === preset.model) ? preset.model : null) || (S.task === "classification" ? "logreg" : "ridge"));
    const methodSel = select([{ value: "kbest", label: "F-test: score each column on its own (fast)" }, { value: "mutual_info", label: "mutual information: catches curved links" }, { value: "rfe", label: "recursive elimination: drop the weakest, repeat" }, { value: "model", label: "random forest importance" }], "kbest");
    const kInput = el("input", { type: "number", min: 1, max: 200, placeholder: "half of the columns" });
    const host = el("div");
    const run = button("Pick the best columns", { kind: "primary", icon: "list", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("select", id, { ...baseParams(), method: methodSel.value, model: modelSel.value, k: kInput.value ? Number(kInput.value) : undefined }, host, signal); showResult(selectResult(job.result), nextSteps(reg, modelSel.value)); }
      catch (e) { console.error(e); } finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, el("p", { class: "plain", text: "Keeps only the k strongest columns (after cleaning and encoding) and scores the model with and without the rest. Selection happens inside the pipeline, so test rows never influence which columns are kept." }),
      el("div", { class: "grid grid-3" }, field("How to rank columns", methodSel), field("How many to keep (k)", kInput), field("Model to score with", modelSel)), run, host);
  }

  function calibrationForm(reg, preset) {
    const withProba = reg.filter((m) => !["svm"].includes(m.key));
    const modelSel = select(withProba.map((m) => ({ value: m.key, label: m.name })), (preset?.model && withProba.some((m) => m.key === preset.model) ? preset.model : null) || "rf");
    const host = el("div");
    const run = button("Check the confidence", { kind: "primary", icon: "target", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("calibration", id, { ...baseParams(), model: modelSel.value }, host, signal); showResult(calibrationResult(job.result), nextSteps(reg, modelSel.value)); }
      catch (e) { console.error(e); } finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, el("p", { class: "plain", text: "When the model says 80% sure, is it right about 80% of the time? This compares the model as trained with two fixes, sigmoid and isotonic, and tells you whether either helps." }), field("Model", modelSel), run, host);
  }

  function validationResult(r) {
    const pts = r.points.filter((p) => p.test !== null);
    const numericX = pts.every((p) => typeof p.value === "number");
    const xs = numericX ? pts.map((p) => p.value) : pts.map((_, i) => i);
    const logx = numericX && xs[0] > 0 && xs[xs.length - 1] / xs[0] > 50;
    const xv = (i) => (logx ? Math.log10(xs[i]) : xs[i]);
    return [
      el("p", { class: "big-answer", text: r.verdict[0] }),
      el("ul", { class: "verdict" }, r.verdict.slice(1).map((t) => el("li", { text: t }))),
      el("div", { class: "chart-wrap" }, lineChart([
        { name: "training rows", points: pts.map((p, i) => [xv(i), p.train]) },
        { name: "new rows", points: pts.map((p, i) => [xv(i), p.test]), band: pts.map((p, i) => [xv(i), p.test - (p.test_std || 0), p.test + (p.test_std || 0)]) },
      ], { xlabel: r.param + (logx ? " (log scale)" : ""), ylabel: METRIC_NAMES[r.scoring] || r.scoring, height: 300, xfmt: (v) => numericX ? (logx ? fmtv(Math.pow(10, v)) : fmtv(v)) : String(pts[Math.round(v)]?.value ?? "") })),
      el("p", { class: "chart-note", text: "Left: simple. Right: complex. Where both lines are low, the model underfits. Where the top line keeps rising but the lower one falls, it overfits. The best setting is at the peak of the lower line." }),
      details("Every value tried", table(["Setting", { label: "Training score", key: "train", num: true }, { label: "Score on new rows", key: "test", num: true }, { label: "Varies by", key: "test_std", num: true }], pts.map((p) => ({ ...p, Setting: String(p.value) })))),
      r.param_help ? el("p", { class: "small muted", text: `${r.param}: ${r.param_help}` }) : null,
      details("How this was done, step by step", narration(r.narration), true), fullCode(r),
    ];
  }

  function explainResult(r) {
    const parts = [
      el("p", { class: "big-answer", text: `${r.model_name}: what it relies on` }),
      el("p", { class: "muted small", text: `The number tracked is the ${r.tracked}. Measured on ${r.rows_used} rows.` }),
      el("h4", { text: "Which columns the model relies on" }), barChart(r.reliance.map((x) => ({ label: x.feature, value: x.value })), { digits: 4 }),
      el("p", { class: "chart-note", text: "Each column was shuffled on its own and the change in the prediction measured. Longer bar = the model leans on that column more." }),
      el("h4", { text: "How the prediction moves with one column" }),
      el("div", { class: "grid grid-2" }, r.partial_dependence.map((pd) => {
        const xs = pd.numeric ? pd.grid : pd.grid.map((_, i) => i);
        const series = [];
        (pd.ice || []).forEach((line) => series.push({ thin: true, color: "#8a94a6", points: line.map((v, i) => [xs[i], v]).filter((p) => p[1] !== null) }));
        series.push({ name: "average", points: pd.average.map((v, i) => [xs[i], v]).filter((p) => p[1] !== null) });
        return el("div", { class: "col-card" }, el("h4", { text: pd.feature }), lineChart(series, { xlabel: pd.feature, ylabel: r.tracked.split(" ")[0], height: 240, xfmt: (v) => pd.numeric ? fmtv(v) : String(pd.grid[Math.round(v)] ?? "") }), el("p", { class: "chart-note", text: pd.text + (pd.ice?.length ? " Thin grey lines are single rows; if they fan out or cross, this column's effect depends on other columns." : "") }));
      })),
    ];
    if (r.local) {
      parts.push(el("h4", { text: `Why this row got "${r.local.prediction}"` }), el("p", { class: "small", text: r.local.text }),
        barChart(r.local.contributions.map((c) => ({ label: `${c.feature} = ${fmtv(c.value)}`, value: c.effect, color: (c.effect || 0) >= 0 ? "#1a7f4b" : "#c62828" })), { digits: 4 }),
        el("p", { class: "chart-note", text: "Green pushed the prediction up, red pulled it down, compared with a typical value for that column." }));
    }
    parts.push(details("How this was done, step by step", narration(r.narration), true), fullCode(r));
    return parts;
  }

  function selectResult(r) {
    const scoreKey = r.ranking.length && r.ranking[0].rank !== undefined ? "rank" : "score";
    return [
      el("p", { class: "big-answer", text: `${r.selected.length} of ${r.n_features} columns kept` }),
      el("p", { class: "plain", text: r.verdict }),
      el("div", { class: "tiles" }, tile(`All ${r.n_features} columns`, score(r.cv_all)), tile(`Best ${r.k} columns`, score(r.cv_selected))),
      el("h4", { text: "Columns kept" }), el("div", { class: "chips" }, r.selected.map((c) => el("span", { class: "chip on", text: c }))),
      r.dropped.length ? details(`Columns dropped (${r.dropped.length})`, el("div", { class: "chips" }, r.dropped.map((c) => el("span", { class: "chip", text: c })))) : null,
      r.ranking.length ? el("div", {}, el("h4", { text: scoreKey === "rank" ? "Ranking (1 is best)" : "Column scores" }), barChart(r.ranking.map((x) => ({ label: x.feature, value: scoreKey === "rank" ? 1 / x.rank : x.score })), { digits: scoreKey === "rank" ? 2 : 3 })) : null,
      details("How this was done, step by step", narration(r.narration), true), fullCode(r),
    ];
  }

  function calibrationResult(r) {
    const names = Object.keys(r.curves);
    const colors = { "as trained": "#2457d6", "sigmoid (Platt)": "#e8590c", "isotonic": "#1a7f4b" };
    return [
      el("p", { class: "big-answer", text: r.verdict[0] }),
      el("ul", { class: "verdict" }, r.verdict.slice(1).map((t) => el("li", { text: t }))),
      el("div", { class: "tiles" }, names.map((n) => el("div", { class: "tile " + (n === r.best ? "primary" : "") }, el("div", { class: "tile-label", text: n }), el("div", { class: "tile-value", text: `ECE ${score(r.scores[n].ece)}` }), el("div", { class: "small muted", text: `Brier ${score(r.scores[n].brier)}` })))),
      el("div", { class: "chart-wrap" }, lineChart(names.map((n) => ({ name: n, color: colors[n], points: r.curves[n].map((p) => [p.confidence, p.actual]) })), { xd: [0, 1], yd: [0, 1], diagonal: true, xlabel: "what the model says", ylabel: "how often it was right", height: 320 })),
      el("p", { class: "chart-note", text: "The dotted line is perfect calibration. Below it the model is overconfident; above it, underconfident. ECE is the average gap; lower is better." }),
      details("How this was done, step by step", narration(r.narration), true), fullCode(r),
    ];
  }

  // ---- Results ---------------------------------------------------------------
  function autoResult(r) {
    const reg = catalog[S.task];
    const ok = r.leaderboard.filter((x) => x.status === "ok");
    const items = r.leaderboard.map((x) => ({ label: x.name, value: x.status === "ok" ? x.mean : null, err: x.std, note: x.status === "skipped" ? "skipped" : x.status === "error" ? "failed" : "", color: x.family === "ensemble" ? "#e8590c" : famColor(x.family) }));
    const f = r.final; const ev = f.evaluation;
    const parts = [
      el("p", { class: "big-answer", text: `Best model: ${r.best.name}` }),
      el("p", { class: "plain", text: `${METRIC_NAMES[r.scoring] || r.scoring} ${score(r.best.mean)} in cross-validation, ${score(ev.metrics[ev.primary])} on rows it never saw.` }),
      el("ul", { class: "verdict" }, r.verdict.map((t) => el("li", { text: t }))),
      (r.refinements || []).length ? el("div", { class: "refine" }, el("h4", { text: "Refinements tested on the winner" }), r.refinements.map((f) => el("div", { class: "refine-row " + (f.applied ? "on" : "") },
        el("span", { class: "badge " + (f.applied ? "ok" : ""), text: f.applied ? "applied" : "not needed" }),
        el("div", {}, el("strong", { text: f.step }), f.before !== undefined && f.before !== null && f.after !== undefined && f.after !== null ? el("span", { class: "muted small", text: `  ${score(f.before)} to ${score(f.after)}` }) : null, el("p", { class: "small", text: f.text }), f.code ? codeBlock(f.code) : null)))) : null,
      r.balance ? notice("info", "A rare class was detected, so models that support it weigh it more. Balanced accuracy is the fairer number to watch.") : null,
      r.log_target ? notice("info", "The answer column has a long tail, so models predict log(value) and convert back.") : null,
      r.subsampled ? notice("info", `To stay fast, the comparison used ${r.rows_used.toLocaleString()} of your ${r.rows_total.toLocaleString()} rows.`) : null,
      el("h4", { text: `All ${ok.length} models, best first (orange bars are combined models)` }),
      barChart(items, { title: "Ranking" }),
      el("p", { class: "chart-note", text: "Longer bar is better. The small dark line shows how much the score moved between test rounds; models whose lines overlap are effectively tied." }),
      details("Full table", table(["Model", "Type", "Score", "Varies by", "Time", ""], r.leaderboard.map((x, i) => [
        el("span", { class: i === 0 && x.status === "ok" ? "winner" : "" }, x.name), x.family === "ensemble" ? "combined: " + (x.members || []).join(", ") : famName(x.family),
        x.status === "ok" ? score(x.mean) : el("span", { class: "muted", text: x.status === "skipped" ? "skipped" : "failed" }),
        x.status === "ok" ? `± ${x.std.toFixed(3)}` : "", x.status === "ok" ? `${x.seconds}s` : (x.note || ""),
        x.status === "ok" && reg.some((m) => m.key === x.key) ? button("Train this", { kind: "small", onclick: () => { clear(s3); s3.append(step("", `Train ${x.name}`, trainForm(reg, { model: x.key }))); s3.scrollIntoView({ behavior: "smooth" }); } }) : ""]))),
      el("h4", { text: `${r.best.name} in detail` }),
    ];
    const diag = ev.fit_diagnosis || {};
    const diagClass = diag.label === "good fit" ? "diag-good" : diag.label?.includes("overfit") ? "diag-mid" : "diag-bad";
    const verdict = { "good fit": "Good: the model works about as well on new rows as on the ones it learned from.", overfitting: "Careful: the model memorised its training rows and does worse on new ones.", "slight overfitting": "Mostly fine: a small drop on new rows.", underfitting: "Weak: the model is too simple for this data." }[diag.label] || diag.text;
    parts.push(...((r.preprocessing?.notes || []).map((t) => notice("warn", t))));
    parts.push(el("p", {}, el("strong", { class: diagClass, text: `${verdict} ` }), el("span", { class: "muted small", text: `(training rows ${score(ev.train_score)}, test rows ${score(ev.metrics[ev.primary])})` })));
    parts.push(metricTiles(ev.metrics, ev.how_to_read, ev.primary), el("p", { class: "muted small", text: "Hover a box to see what the number means." }));
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
    if (r.model_id) {
      const slot = el("div", { class: "try-slot" }, el("h4", { text: "Try it: enter values and get a prediction" }));
      tryModelPanel(r.model_id, { compact: true }).then((p) => slot.append(p));
      parts.push(slot, el("div", { class: "row" },
        el("a", { class: "btn", href: `/api/models/${r.model_id}/download`, download: "" }, svgIcon("download"), el("span", { text: "Download model" })),
        el("a", { class: "btn", href: `/api/models/${r.model_id}/notebook`, download: "" }, svgIcon("book"), el("span", { text: "Download notebook" })),
        el("a", { class: "btn", href: `#/dataset/${id}/predict` }, svgIcon("upload"), el("span", { text: "Predict a whole file" }))));
    }
    parts.push(details("How this was done, step by step", narration(r.narration), true));
    if (r.models_tried) parts.push(details("Every model that was tried, and how each one was set up", el("div", { class: "stack" }, r.models_tried.map((m) => el("div", { class: "col-card" },
      el("div", { class: "row" }, el("strong", { text: m.name }), el("span", { class: "badge", text: famName(m.family) }), m.mean !== null && m.mean !== undefined ? el("span", { class: "badge ok", text: score(m.mean) }) : el("span", { class: "badge", text: m.status })),
      el("p", { class: "small", text: m.explain }), codeBlock(m.code))))));
    parts.push(details("How the data was prepared", prepSteps(r.preprocessing)), fullCode(r));
    const bestKey = ok.find((x) => x.family !== "ensemble")?.key;
    const wrap = el("div", {}, ...parts);
    setTimeout(() => s4.append(nextSteps(reg, bestKey)), 0);
    return wrap;
  }

  function raceResult(r, reg) {
    const ok = r.leaderboard.filter((x) => x.status === "ok");
    const items = r.leaderboard.map((x) => ({ label: x.name, value: x.status === "ok" ? x.mean : null, err: x.std, note: x.status === "skipped" ? "skipped" : x.status === "error" ? "failed" : "", color: famColor(x.family) }));
    const rows = r.leaderboard.map((x, i) => [
      el("span", { class: i === 0 && x.status === "ok" ? "winner" : "" }, x.name), famName(x.family),
      x.status === "ok" ? x.mean.toFixed(4) : el("span", { class: "muted", text: x.status === "skipped" ? "skipped" : "failed" }),
      x.status === "ok" ? `± ${x.std.toFixed(4)}` : "", x.status === "ok" ? `${x.seconds}s` : (x.note || ""),
      x.status === "ok" && reg.some((m) => m.key === x.key) ? button("Train this", { kind: "small", onclick: () => { clear(s3); s3.append(step("", `Train ${x.name}`, trainForm(reg, { model: x.key }))); s3.scrollIntoView({ behavior: "smooth" }); } }) : "",
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
      ...((r.preprocessing?.notes || []).map((t) => notice("warn", t))),
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
    if (r.model_id) {
      const slot = el("div", { class: "try-slot" }, el("h4", { text: "Try it: enter values and get a prediction" }));
      tryModelPanel(r.model_id, { compact: true }).then((p) => slot.append(p));
      parts.push(slot, el("div", { class: "row" },
        el("a", { class: "btn", href: `/api/models/${r.model_id}/download`, download: "" }, svgIcon("download"), el("span", { text: "Download model" })),
        el("a", { class: "btn", href: `/api/models/${r.model_id}/notebook`, download: "" }, svgIcon("book"), el("span", { text: "Download notebook" })),
        el("a", { class: "btn", href: `#/dataset/${id}/predict` }, svgIcon("upload"), el("span", { text: "Predict a whole file" }))));
    }
    parts.push(details("How this was done, step by step", narration(r.narration), true), details("How the data was prepared", prepSteps(r.preprocessing)), fullCode(r));
    return parts;
  }

  function tuneResult(r) {
    const best = r.trials[0];
    const gain = (best.mean || 0) - r.baseline;
    const scored = r.trials.filter((t) => t.mean !== null);
    const items = scored.slice(0, 10).map((t) => ({ label: Object.entries(t.params).map(([k, v]) => `${k}=${fmtv(v)}`).join(", "), value: t.mean, err: t.std }));
    items.push({ label: "default settings", value: r.baseline, color: "#cbd3de" });
    const keys = Object.keys(best.params);
    const trialTable = table([{ label: "#", get: (t) => t.rank }, ...keys.map((k) => ({ label: k, num: true, get: (t) => fmtv(t.params[k]) })), { label: "score", num: true, get: (t) => score(t.mean) }],
      scored.map((t, i) => ({ ...t, rank: i + 1 })));
    return [
      el("p", { class: "big-answer", text: gain > 0.005 ? `Tuning improved ${r.model_name} from ${score(r.baseline)} to ${score(best.mean)}` : `Tuning did not help much: ${score(r.baseline)} to ${score(best.mean)}. The defaults were already good.` }),
      el("div", { class: "chart-wrap" }, barChart(items, { digits: 4 })),
      el("p", { class: "chart-note", text: scored.length > 10 ? `The ${Math.min(10, scored.length)} best of ${scored.length} settings tried, best first. Grey is the untouched default. The full list is below.` : "Each bar is one set of settings, best first. Grey is the untouched default." }),
      details(`Every setting tried (${scored.length})`, trialTable),
      r.search === "grid" && r.grid ? el("p", { class: "small", text: "Grid tried: " + Object.entries(r.grid).map(([k, v]) => `${k} in [${v.map(fmtv).join(", ")}]`).join("; ") + `. ${r.trials.length} combinations.` }) : null,
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

function planSummary(prep) {
  const bits = [];
  if (prep.drop_columns?.length) bits.push(`leave out ${prep.drop_columns.join(", ")}`);
  if (prep.drop_duplicates) bits.push("remove duplicates");
  bits.push(`fill blanks (${prep.num_impute === "median" ? "median" : prep.num_impute.replace("_", " ")})`);
  if (prep.outliers === "cap") bits.push("cap outliers");
  if (prep.log_skewed) bits.push("log-transform long tails");
  if (prep.expand_dates) bits.push("turn dates into numbers");
  bits.push(prep.encoder === "onehot" ? (prep.rare_min > 1 ? "one-hot encode text, grouping rare values" : "one-hot encode text") : "number-code text");
  if (prep.ordered_auto) bits.push("keep the order of words like low/medium/high");
  if (prep.drop_correlated) bits.push("drop near-duplicate columns");
  if (prep.poly_degree) bits.push(prep.poly_interaction_only ? "add pair products" : `add polynomial features (degree ${prep.poly_degree})`);
  if (prep.bin_numeric) bits.push(`cut numbers into ${prep.bin_numeric} bins`);
  if (prep.resample && prep.resample !== "none") bits.push({ oversample: "copy rare rows", undersample: "drop common rows", smote: "create rare rows with SMOTE" }[prep.resample]);
  else if (prep.balance !== "off") bits.push("weigh a rare class more");
  if (prep.scaler !== "none") bits.push("scale numbers");
  return bits.join(", ");
}

function tile(label, value) { return el("div", { class: "tile" }, el("div", { class: "tile-label", text: label }), el("div", { class: "tile-value", text: String(value) })); }

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

function famColor(f) { return { linear: "#2457d6", distance: "#0e8ea8", probabilistic: "#6f42c1", tree: "#5c940d", kernel: "#c2255c", neural: "#7c4a1e", bagging: "#1a7f4b", boosting: "#8a5a00" }[f] || "#8a94a6"; }
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
