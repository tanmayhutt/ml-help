import { api, cached } from "../api.js";
import { el, clear, card, button, notice, details, svgIcon, tabs, codeBlock, fullCode } from "../ui.js";
import { runJob } from "../jobrun.js";

export async function render(root, [lessonKey], signal) {
  const [glossary, lessons, catalog] = await Promise.all([cached("glossary"), cached("lessons"), cached("catalog")]);
  root.append(el("div", { class: "page-head" }, el("div", { class: "grow" },
    el("h1", { text: "Learn" }),
    el("p", { text: "Guided lessons that run real experiments on bundled data, the full model shelf with explanations, and a glossary." }))));

  const lessonsTab = () => el("div", { class: "stack" }, lessons.map((L) => lessonCard(L, signal)));
  const modelsTab = () => el("div", { class: "stack" },
    el("h3", { text: "Model families" }), el("dl", { class: "glossary" }, Object.entries(catalog.families).flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })])),
    el("h3", { text: "Ensembles" }), el("dl", { class: "glossary" }, Object.entries(catalog.ensembles).flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })])),
    el("h3", { text: "Classification models" }), modelList(catalog.classification),
    el("h3", { text: "Regression models" }), modelList(catalog.regression),
    el("h3", { text: "Unsupervised" }), el("dl", { class: "glossary" }, Object.entries(catalog.unsupervised).flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })])));
  const glossaryTab = () => el("dl", { class: "glossary" }, glossary.flatMap((g) => [el("dt", { text: g.term }), el("dd", { text: g.text })]));
  const howTab = () => el("div", { class: "stack" },
    el("p", { text: "ML Help runs scikit-learn on the server, one job at a time, inside a throwaway process with a time and memory cap. Nothing runs in your browser except this page." }),
    el("ol", {}, [
      "Upload: the file is parsed with pandas, profiled, and stored as a pickle for fast reloads.",
      "Preprocess: a ColumnTransformer imputes, scales and encodes. It is fitted inside the pipeline so it only ever sees training rows.",
      "Race: each model is cross-validated on a capped sample; slow models are skipped on big data unless you insist.",
      "Train: one hold-out split, full metrics, charts, importance, a saved pipeline and a notebook that reproduces it.",
      "Tune: random search with at most 20 trials. Learning curves use five training sizes.",
      "Unsupervised: clustering with an automatic k sweep, PCA with loadings, t-SNE on a capped sample, isolation forest and LOF.",
    ].map((t) => el("li", { text: t }))),
    el("p", { class: "muted small", text: "Why the caps: the server is shared with other apps. The caps keep every job to seconds and the machine cost flat." }));
  root.append(card(null, tabs([{ label: "Lessons", render: lessonsTab }, { label: "Model shelf", render: modelsTab }, { label: "Glossary", render: glossaryTab }, { label: "How this works", render: howTab }], lessonKey === "models" ? 1 : lessonKey === "glossary" ? 2 : 0)));
}

function modelList(models) {
  return el("div", { class: "grid grid-2" }, models.map((m) => el("div", { class: "col-card" },
    el("div", { class: "row" }, el("strong", { text: m.name }), el("span", { class: "badge", text: m.family }), el("span", { class: "badge", text: `cost ${m.cost}/5` })),
    el("p", { class: "small", text: m.explain }),
    Object.keys(m.params).length ? el("dl", { class: "glossary small" }, Object.entries(m.params).flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })])) : null)));
}

function lessonCard(L, signal) {
  const body = el("div", { class: "stack" }, el("p", { text: L.intro }));
  let datasetId = null;
  const steps = L.steps.map((st, i) => {
    const host = el("div");
    const out = el("div");
    const run = button("Run", { kind: "small primary", icon: "play", onclick: async () => {
      run.disabled = true;
      try {
        if (!datasetId) datasetId = (await api.loadSample(L.sample)).id;
        const job = await runJob(st.job.kind, datasetId, st.job.params, host, signal);
        clear(out).append(lessonSummary(job.result, st.job.kind), details("Python code for this step", codeBlock(job.result.code || "")));
      } catch {} finally { run.disabled = false; }
    } });
    return el("div", { class: "lesson-step" }, el("div", { class: "n", text: i + 1 }), el("div", { class: "grow" }, el("p", { text: st.text }), host, out), run);
  });
  body.append(...steps, el("p", {}, el("strong", { text: "Takeaway. " }), L.takeaway), el("p", { class: "muted small" }, "Open the full dataset page for every chart: ", el("a", { href: "#/datasets", text: "Datasets" })));
  return details(`${L.title} (${L.sample})`, body);
}

function lessonSummary(r, kind) {
  if (kind === "leaderboard") return el("div", { class: "stack" }, el("ul", {}, r.leaderboard.filter((x) => x.status === "ok").map((x) => el("li", { text: `${x.name}: ${x.mean.toFixed(4)}` }))), ...r.summary.map((t) => el("p", { class: "small", text: t })));
  if (kind === "train") { const ev = r.evaluation; return el("p", { class: "small", text: `${r.model_name}: train ${ev.primary} ${ev.train_score}, test ${ev.metrics[ev.primary]}. ${ev.fit_diagnosis.label}. ${ev.fit_diagnosis.text}` }); }
  if (kind === "tune") return el("p", { class: "small", text: r.narration.map((n) => n.text).join(" ") });
  if (kind === "curve") return el("p", { class: "small", text: r.verdict });
  if (kind === "cluster") return el("p", { class: "small", text: `${r.evaluation.n_clusters} clusters, silhouette ${r.evaluation.metrics.silhouette ?? "n/a"}${r.suggested_k ? `, data suggests k=${r.suggested_k}` : ""}${r.evaluation.noise ? `, ${r.evaluation.noise} noise rows` : ""}.` });
  if (kind === "reduce") return el("p", { class: "small", text: r.cumulative ? `Two components explain ${(r.cumulative[1] * 100).toFixed(0)}% of the variance.` : "Done." });
  return el("p", { class: "small", text: "Done." });
}
