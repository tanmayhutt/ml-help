import { api, cached } from "../api.js";
import { el, clear, card, button, notice, table, select, field, details, codeBlock, metricTiles, narration, tabs, svgIcon, fullCode } from "../ui.js";
import { scatterChart, lineChart, barChart } from "../charts.js";
import { runJob } from "../jobrun.js";
import { prepSteps, chip } from "./experiment.js";

export async function render(root, [id], signal) {
  const [ds, catalog] = await Promise.all([api.dataset(id), cached("catalog")]);
  const cols = ds.profile.columns;
  const S = { prep: { scaler: "standard", encoder: "onehot", num_impute: "median", cat_impute: "most_frequent", drop_columns: [] } };
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("div", { class: "crumb" }, el("a", { href: "#/datasets", text: "Datasets" }), svgIcon("arrow", 12), el("a", { href: `#/dataset/${id}`, text: ds.name }), svgIcon("arrow", 12), el("span", { text: "Unsupervised" })),
    el("h1", { text: "Unsupervised learning" }),
    el("p", { text: "No target column. Find groups, compress the columns into a 2D map, or flag unusual rows." }))));

  const dropChips = el("div", { class: "chips" }, cols.map((c) => chip(c.name, false, (on) => { S.prep.drop_columns = on ? [...new Set([...S.prep.drop_columns, c.name])] : S.prep.drop_columns.filter((x) => x !== c.name); })));
  root.append(card("Columns and scaling", [
    el("p", { class: "muted small", text: "Exclude label columns or IDs so the algorithms only see real features. Scaling is on by default because every method here measures distances." }),
    el("div", { class: "grid grid-3" }, field("Scale numbers", select(["standard", "minmax", "robust", "none"], "standard", (v) => (S.prep.scaler = v))), field("Encode text", select(["onehot", "ordinal"], "onehot", (v) => (S.prep.encoder = v)))),
    details("Exclude columns", dropChips, true)]));

  const results = el("div");
  const colorOpts = [{ value: "", label: "none" }, ...cols.map((c) => ({ value: c.name, label: c.name }))];

  function clusterTab() {
    let algo = "kmeans", k = 3, eps = 0.5, ms = 5, linkage = "ward";
    const explain = el("p", { class: "muted small", text: catalog.unsupervised.kmeans });
    const params = el("div", { class: "grid grid-3" });
    const algoSel = select([{ value: "kmeans", label: "K-Means" }, { value: "dbscan", label: "DBSCAN" }, { value: "agglomerative", label: "Agglomerative" }, { value: "gmm", label: "Gaussian Mixture" }], algo, (v) => { algo = v; explain.textContent = catalog.unsupervised[v]; refresh(); });
    function refresh() {
      clear(params);
      if (algo === "dbscan") params.append(field("eps", el("input", { type: "number", step: 0.1, min: 0.05, value: eps, oninput: (e) => (eps = Number(e.target.value)) }), "Neighbourhood radius in scaled units."), field("min_samples", el("input", { type: "number", min: 2, value: ms, oninput: (e) => (ms = Number(e.target.value)) }), "Neighbours needed to form a core point."));
      else params.append(field("k (clusters)", el("input", { type: "number", min: 2, max: 20, value: k, oninput: (e) => (k = Number(e.target.value)) })));
      if (algo === "agglomerative") params.append(field("linkage", select(["ward", "complete", "average", "single"], linkage, (v) => (linkage = v)), "How the distance between two clusters is measured."));
    }
    refresh();
    const host = el("div");
    const run = button("Cluster", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try {
        const job = await runJob("cluster", id, { algorithm: algo, k, eps, min_samples: ms, linkage, preprocess: S.prep }, host, signal);
        clear(results).append(clusterResult(job.result));
        results.scrollIntoView({ behavior: "smooth" });
      } catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, field("Algorithm", algoSel), explain, params, run, host);
  }

  function reduceTab() {
    let algo = "pca", n = 2, perp = 30, colorBy = "";
    const explain = el("p", { class: "muted small", text: catalog.unsupervised.pca });
    const algoSel = select([{ value: "pca", label: "PCA" }, { value: "tsne", label: "t-SNE" }], algo, (v) => { algo = v; explain.textContent = catalog.unsupervised[v]; });
    const host = el("div");
    const run = button("Project to 2D", { kind: "primary", icon: "scatter", onclick: async () => {
      run.disabled = true;
      try {
        const job = await runJob("reduce", id, { algorithm: algo, n_components: n, perplexity: perp, color_by: colorBy || null, preprocess: S.prep }, host, signal);
        clear(results).append(reduceResult(job.result));
        results.scrollIntoView({ behavior: "smooth" });
      } catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, el("div", { class: "grid grid-3" },
      field("Algorithm", algoSel), field("Colour points by", select(colorOpts, "", (v) => (colorBy = v)), "A label column, if you have one, to see whether the structure matches it."),
      field("PCA components", el("input", { type: "number", min: 2, max: 10, value: n, oninput: (e) => (n = Number(e.target.value)) })),
      field("t-SNE perplexity", el("input", { type: "number", min: 5, max: 50, value: perp, oninput: (e) => (perp = Number(e.target.value)) }))), explain, run, host);
  }

  function anomalyTab() {
    let algo = "isoforest", cont = 0.05;
    const explain = el("p", { class: "muted small", text: catalog.unsupervised.isoforest });
    const host = el("div");
    const run = button("Find anomalies", { kind: "primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try {
        const job = await runJob("anomaly", id, { algorithm: algo, contamination: cont, preprocess: S.prep }, host, signal);
        clear(results).append(anomalyResult(job.result));
        results.scrollIntoView({ behavior: "smooth" });
      } catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "stack" }, el("div", { class: "grid grid-3" },
      field("Algorithm", select([{ value: "isoforest", label: "Isolation Forest" }, { value: "lof", label: "Local Outlier Factor" }], algo, (v) => { algo = v; explain.textContent = catalog.unsupervised[v]; })),
      field("Expected share of anomalies", el("input", { type: "number", min: 0.005, max: 0.3, step: 0.01, value: cont, oninput: (e) => (cont = Number(e.target.value)) }))), explain, run, host);
  }

  root.append(card("Run", tabs([{ label: "Clustering", render: clusterTab }, { label: "Dimensionality reduction", render: reduceTab }, { label: "Anomaly detection", render: anomalyTab }])));
  root.append(results);

  function clusterResult(r) {
    const ev = r.evaluation;
    const parts = [
      el("p", { text: r.explain }),
      el("div", { class: "row" }, el("span", { class: "badge accent", text: `${ev.n_clusters} clusters` }), ev.noise ? el("span", { class: "badge warn", text: `${ev.noise} noise rows` }) : null, el("span", { class: "muted small", text: `${r.rows_used.toLocaleString()} rows` })),
      Object.keys(ev.metrics).length ? metricTiles(ev.metrics, ev.how_to_read, "silhouette") : notice("info", "Cluster quality metrics need at least two clusters."),
      el("div", { class: "grid grid-2" },
        el("div", { class: "col-card" }, el("h4", { text: "Clusters in 2D" }), scatterChart(r.points.xy, r.points.labels), el("p", { class: "chart-note", text: r.points.axis })),
        el("div", { class: "col-card" }, el("h4", { text: "Cluster sizes" }), barChart(r.sizes.map((s) => ({ label: s.label === -1 ? "noise" : `cluster ${s.label}`, value: s.count })), { digits: 0 }))),
    ];
    if (r.sweep) {
      parts.push(el("div", { class: "grid grid-2" },
        el("div", { class: "col-card" }, el("h4", { text: "Elbow: inertia vs k" }), lineChart([{ name: "inertia", points: r.sweep.map((s) => [s.k, s.inertia]) }], { xlabel: "k", ylabel: "inertia", height: 220, xfmt: (v) => Math.round(v) }), el("p", { class: "chart-note", text: "Look for the bend where the curve flattens." })),
        el("div", { class: "col-card" }, el("h4", { text: "Silhouette vs k" }), lineChart([{ name: "silhouette", points: r.sweep.filter((s) => s.silhouette !== null).map((s) => [s.k, s.silhouette]) }], { xlabel: "k", ylabel: "silhouette", height: 220, xfmt: (v) => Math.round(v) }), el("p", { class: "chart-note", text: `Peak at k = ${r.suggested_k}.` }))));
    }
    parts.push(details("Step by step", narration(r.narration), true), details("Preprocessing", prepSteps(r.preprocessing)), fullCode(r));
    return card(`${r.algorithm} result`, parts);
  }

  function reduceResult(r) {
    const parts = [el("div", { class: "col-card" }, el("h4", { text: "2D map" }), scatterChart(r.points.xy, r.points.labels, { height: 420 }), el("p", { class: "chart-note", text: r.points.axis }))];
    if (r.explained) {
      parts.push(el("div", { class: "grid grid-2" },
        el("div", { class: "col-card" }, el("h4", { text: "Explained variance" }), lineChart([{ name: "per component", points: r.explained.map((v, i) => [i + 1, v]) }, { name: "cumulative", points: r.cumulative.map((v, i) => [i + 1, v]) }], { xlabel: "component", ylabel: "share of variance", yd: [0, 1], height: 220, xfmt: (v) => Math.round(v) })),
        el("div", { class: "col-card" }, el("h4", { text: "What PC1 and PC2 are made of" }), ...r.loadings.map((l, i) => el("div", {}, el("strong", { class: "small", text: `PC${i + 1}` }), barChart(l.map((x) => ({ label: x.feature, value: x.weight })), { digits: 3 }))))));
    }
    parts.push(details("Step by step", narration(r.narration), true), fullCode(r));
    return card(`${r.algorithm.toUpperCase()} result`, parts);
  }

  function anomalyResult(r) {
    return card(`${r.flagged} anomalies flagged`, [
      el("div", { class: "col-card" }, el("h4", { text: "Flagged rows in 2D (1 = anomaly)" }), scatterChart(r.points.xy, r.points.labels), el("p", { class: "chart-note", text: r.points.axis })),
      el("h4", { text: "Most anomalous rows" }), table(r.columns, r.top, { class: "compact" }),
      details("Step by step", narration(r.narration), true), fullCode(r)]);
  }
}
