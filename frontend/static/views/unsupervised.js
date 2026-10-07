import { api, cached } from "../api.js";
import { el, clear, step, button, notice, table, select, field, details, metricTiles, narration, svgIcon, choices, fullCode } from "../ui.js";
import { scatterChart, lineChart, barChart } from "../charts.js";
import { runJob } from "../jobrun.js";
import { prepSteps, chip } from "./experiment.js";

export async function render(root, [id], signal) {
  const [ds, catalog] = await Promise.all([api.dataset(id), cached("catalog")]);
  const cols = ds.profile.columns;
  const S = { prep: { scaler: "standard", encoder: "onehot", num_impute: "median", cat_impute: "most_frequent", drop_columns: [] } };
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("div", { class: "crumb" }, el("a", { href: "#/datasets", text: "Home" }), svgIcon("arrow", 12), el("a", { href: `#/dataset/${id}`, text: ds.name }), svgIcon("arrow", 12), el("span", { text: "Find groups or odd rows" })),
    el("h1", { text: "Find groups or odd rows" }))));

  const s2 = el("div"), s3 = el("div");
  const dropChips = el("div", { class: "chips" }, cols.map((c) => chip(c.name, false, (on) => { S.prep.drop_columns = on ? [...new Set([...S.prep.drop_columns, c.name])] : S.prep.drop_columns.filter((x) => x !== c.name); })));
  root.append(step(1, "Which columns should be used?", [
    el("p", { class: "plain", text: "All columns are used unless you untick them. Leave out IDs, names, and any answer column." }),
    dropChips,
    details("Settings", el("div", { class: "grid grid-3" }, field("Scale numbers", select([{ value: "standard", label: "yes (recommended)" }, { value: "none", label: "no" }], "standard", (v) => (S.prep.scaler = v))))),
    button("Continue", { kind: "primary", icon: "arrow", onclick: buildStep2 })]), s2, s3);

  function buildStep2() {
    clear(s2); clear(s3);
    const form = el("div");
    s2.append(step(2, "What do you want to find?", [choices([
      { key: "cluster", icon: "layers", title: "Groups of similar rows", text: "Rows that look alike are put in the same group." },
      { key: "reduce", icon: "scatter", title: "A 2D picture of the data", text: "Squeeze all columns into a dot plot you can look at." },
      { key: "anomaly", icon: "warn", title: "Unusual rows", text: "Find rows that do not look like the rest." },
    ], (k) => clear(form).append({ cluster: clusterForm, reduce: reduceForm, anomaly: anomalyForm }[k]())), form]));
    s2.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function showResult(node) { clear(s3).append(step(3, "Result", node)); s3.scrollIntoView({ behavior: "smooth", block: "start" }); }

  function clusterForm() {
    let algo = "kmeans", k = 3, eps = 0.5, ms = 5;
    const params = el("div", { class: "grid grid-3" });
    const algoSel = select([{ value: "kmeans", label: "K-Means: I choose how many groups (recommended)" }, { value: "dbscan", label: "DBSCAN: find groups of any shape, mark leftovers" }, { value: "agglomerative", label: "Agglomerative: merge closest rows step by step" }, { value: "gmm", label: "Gaussian Mixture: soft, overlapping groups" }], algo, (v) => { algo = v; refresh(); });
    function refresh() {
      clear(params);
      if (algo === "dbscan") params.append(field("How close is 'close' (eps)", el("input", { type: "number", step: 0.1, min: 0.05, value: eps, oninput: (e) => (eps = Number(e.target.value)) })), field("Minimum rows per group", el("input", { type: "number", min: 2, value: ms, oninput: (e) => (ms = Number(e.target.value)) })));
      else params.append(field("How many groups", el("input", { type: "number", min: 2, max: 20, value: k, oninput: (e) => (k = Number(e.target.value)) }), algo === "kmeans" ? "Not sure? Run it; the result suggests a number." : ""));
    }
    refresh();
    const host = el("div");
    const run = button("Find groups", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("cluster", id, { algorithm: algo, k, eps, min_samples: ms, preprocess: S.prep }, host, signal); showResult(clusterResult(job.result)); }
      catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, field("Method", algoSel), params, run, host);
  }

  function reduceForm() {
    let algo = "pca", colorBy = "";
    const host = el("div");
    const run = button("Draw the picture", { kind: "primary", icon: "scatter", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("reduce", id, { algorithm: algo, n_components: 2, perplexity: 30, color_by: colorBy || null, preprocess: S.prep }, host, signal); showResult(reduceResult(job.result)); }
      catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, el("div", { class: "grid grid-3" },
      field("Method", select([{ value: "pca", label: "PCA: fast, keeps big-picture shape" }, { value: "tsne", label: "t-SNE: slower, shows tight clusters" }], algo, (v) => (algo = v))),
      field("Colour the dots by", select([{ value: "", label: "nothing" }, ...cols.map((c) => ({ value: c.name, label: c.name }))], "", (v) => (colorBy = v)))), run, host);
  }

  function anomalyForm() {
    let algo = "isoforest", cont = 0.05;
    const host = el("div");
    const run = button("Find unusual rows", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try { const job = await runJob("anomaly", id, { algorithm: algo, contamination: cont, preprocess: S.prep }, host, signal); showResult(anomalyResult(job.result)); }
      catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, el("div", { class: "grid grid-3" },
      field("Method", select([{ value: "isoforest", label: "Isolation Forest (recommended)" }, { value: "lof", label: "Local Outlier Factor" }], algo, (v) => (algo = v))),
      field("Roughly what share is unusual", select([{ value: 0.02, label: "2%" }, { value: 0.05, label: "5%" }, { value: 0.1, label: "10%" }], 0.05, (v) => (cont = Number(v))))), run, host);
  }

  function clusterResult(r) {
    const ev = r.evaluation;
    const parts = [
      el("p", { class: "big-answer", text: `Found ${ev.n_clusters} groups${ev.noise ? ` and ${ev.noise} rows that fit nowhere` : ""}${r.suggested_k ? `. The data itself suggests ${r.suggested_k} groups.` : "."}` }),
      el("p", { text: r.explain }),
      Object.keys(ev.metrics).length ? metricTiles(ev.metrics, ev.how_to_read, "silhouette") : null,
      el("p", { class: "muted small", text: "Silhouette: closer to 1 means clear, well-separated groups. Hover a box for more." }),
      el("div", { class: "grid grid-2" },
        el("div", { class: "col-card" }, el("h4", { text: "The groups" }), scatterChart(r.points.xy, r.points.labels), el("p", { class: "chart-note", text: "Each dot is a row, colour is its group. " + r.points.axis })),
        el("div", { class: "col-card" }, el("h4", { text: "Rows per group" }), barChart(r.sizes.map((s) => ({ label: s.label === -1 ? "no group" : `group ${s.label}`, value: s.count })), { digits: 0 }))),
    ];
    if (r.sweep) parts.push(el("div", { class: "grid grid-2" },
      el("div", { class: "col-card" }, el("h4", { text: "How many groups? (elbow)" }), lineChart([{ name: "spread", points: r.sweep.map((s) => [s.k, s.inertia]) }], { xlabel: "groups", ylabel: "spread inside groups", height: 220, xfmt: (v) => Math.round(v) }), el("p", { class: "chart-note", text: "Pick the bend where the line stops dropping fast." })),
      el("div", { class: "col-card" }, el("h4", { text: "How many groups? (silhouette)" }), lineChart([{ name: "silhouette", points: r.sweep.filter((s) => s.silhouette !== null).map((s) => [s.k, s.silhouette]) }], { xlabel: "groups", ylabel: "silhouette", height: 220, xfmt: (v) => Math.round(v) }), el("p", { class: "chart-note", text: `Highest point is at ${r.suggested_k} groups.` }))));
    parts.push(details("How this was done, step by step", narration(r.narration), true), details("How the data was prepared", prepSteps(r.preprocessing)), fullCode(r));
    return parts;
  }

  function reduceResult(r) {
    const parts = [el("div", { class: "col-card" }, el("h4", { text: "Your data as a picture" }), scatterChart(r.points.xy, r.points.labels, { height: 420 }), el("p", { class: "chart-note", text: "Each dot is a row. Dots near each other are similar rows. " + r.points.axis }))];
    if (r.explained) parts.push(el("p", { class: "big-answer", text: `This picture keeps ${(r.cumulative[1] * 100).toFixed(0)}% of what makes your rows different.` }),
      details("Which columns shape the picture", el("div", { class: "grid grid-2" }, r.loadings.map((l, i) => el("div", { class: "col-card" }, el("h4", { text: i === 0 ? "Left to right" : "Bottom to top" }), barChart(l.map((x) => ({ label: x.feature, value: x.weight })), { digits: 3 }))))));
    parts.push(details("How this was done, step by step", narration(r.narration), true), fullCode(r));
    return parts;
  }

  function anomalyResult(r) {
    return [
      el("p", { class: "big-answer", text: `${r.flagged} unusual rows found out of ${r.rows_used.toLocaleString()}.` }),
      el("div", { class: "col-card" }, el("h4", { text: "Where they are" }), scatterChart(r.points.xy, r.points.labels), el("p", { class: "chart-note", text: "Orange dots are the unusual rows. " + r.points.axis })),
      el("h4", { text: "Most unusual rows first" }), table(r.columns.map((c) => c === "_score" ? "how unusual" : c), r.top, { class: "compact" }),
      details("How this was done, step by step", narration(r.narration), true), fullCode(r)];
  }
}
