// Hand-drawn SVG charts. Each function returns an <svg> element sized to its container width via viewBox.
const NS = "http://www.w3.org/2000/svg";
const PALETTE = ["#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2", "#be185d", "#4d7c0f", "#b45309", "#1d4ed8"];

function s(tag, attrs = {}, text) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}

function frame(w, h, title) {
  const svg = s("svg", { viewBox: `0 0 ${w} ${h}`, class: "chart", role: "img", "aria-label": title || "chart" });
  svg.style.maxWidth = w + "px";
  if (title) svg.append(s("title", {}, title));
  return svg;
}

function scale(domain, range) {
  const [d0, d1] = domain; const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}

function nice(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(0) + "k";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(2);
}

function axes(svg, x, y, xd, yd, m, w, h, opts = {}) {
  const g = s("g", { class: "axes" });
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const yv = yd[0] + ((yd[1] - yd[0]) * i) / ticks;
    const yy = y(yv);
    g.append(s("line", { x1: m.l, x2: w - m.r, y1: yy, y2: yy, class: "grid" }));
    g.append(s("text", { x: m.l - 8, y: yy + 4, "text-anchor": "end", class: "tick" }, nice(yv)));
  }
  for (let i = 0; i <= ticks; i++) {
    const xv = xd[0] + ((xd[1] - xd[0]) * i) / ticks;
    g.append(s("text", { x: x(xv), y: h - m.b + 18, "text-anchor": "middle", class: "tick" }, opts.xfmt ? opts.xfmt(xv) : nice(xv)));
  }
  if (opts.xlabel) g.append(s("text", { x: (m.l + w - m.r) / 2, y: h - 6, "text-anchor": "middle", class: "axis-label" }, opts.xlabel));
  if (opts.ylabel) g.append(s("text", { x: 12, y: (m.t + h - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(m.t + h - m.b) / 2})`, class: "axis-label" }, opts.ylabel));
  svg.append(g);
}

// Horizontal bars as plain HTML so labels stay readable at any width. items [{label, value, err?, color?, note?}]
export function barChart(items, opts = {}) {
  const wrap = document.createElement("div"); wrap.className = "hbars";
  const vals = items.map((i) => i.value).filter((v) => v !== null && v !== undefined);
  const lo = Math.min(0, ...vals), hi = Math.max(1e-9, ...vals);
  const pct = (v) => ((v - lo) / (hi - lo)) * 100;
  for (const it of items) {
    const row = document.createElement("div"); row.className = "hbar-row";
    const label = document.createElement("div"); label.className = "hbar-label"; label.textContent = it.label; label.title = it.label;
    const track = document.createElement("div"); track.className = "hbar-track";
    const val = document.createElement("div"); val.className = "hbar-value";
    if (it.value === null || it.value === undefined) {
      val.textContent = it.note || "skipped"; val.classList.add("muted");
    } else {
      const bar = document.createElement("div"); bar.className = "hbar";
      bar.style.left = pct(Math.min(0, it.value)) + "%"; bar.style.width = Math.max(Math.abs(pct(it.value) - pct(0)), 0.5) + "%";
      if (it.color) bar.style.background = it.color;
      track.append(bar);
      if (it.err) { const e = document.createElement("div"); e.className = "hbar-err"; e.style.left = pct(it.value - it.err) + "%"; e.style.width = Math.max(pct(it.value + it.err) - pct(it.value - it.err), 0.3) + "%"; track.append(e); }
      val.textContent = it.value.toFixed(opts.digits ?? 3);
    }
    row.append(label, track, val); wrap.append(row);
  }
  return wrap;
}

// Lines: series [{name, points:[[x,y]], color?, band?:[[x,lo,hi]]}]
export function lineChart(series, opts = {}) {
  const w = 560, h = opts.height || 300, m = { l: 56, r: 16, t: 16, b: 44 };
  const svg = frame(w, h, opts.title);
  const xs = series.flatMap((sr) => sr.points.map((p) => p[0]));
  const ys = series.flatMap((sr) => sr.points.map((p) => p[1]));
  const xd = opts.xd || [Math.min(...xs), Math.max(...xs)];
  const yd = opts.yd || [Math.min(...ys), Math.max(...ys)];
  if (yd[0] === yd[1]) { yd[0] -= 0.5; yd[1] += 0.5; }
  const x = scale(xd, [m.l, w - m.r]), y = scale(yd, [h - m.b, m.t]);
  axes(svg, x, y, xd, yd, m, w, h, opts);
  if (opts.diagonal) svg.append(s("line", { x1: x(xd[0]), y1: y(yd[0]), x2: x(xd[1]), y2: y(yd[1]), class: "diag" }));
  series.forEach((sr, i) => {
    const c = sr.color || PALETTE[i % PALETTE.length];
    if (sr.band) {
      const up = sr.band.map((b) => `${x(b[0])},${y(b[2])}`), lo = sr.band.slice().reverse().map((b) => `${x(b[0])},${y(b[1])}`);
      svg.append(s("polygon", { points: [...up, ...lo].join(" "), fill: c, opacity: 0.12 }));
    }
    svg.append(s("polyline", { points: sr.points.map((p) => `${x(p[0])},${y(p[1])}`).join(" "), fill: "none", stroke: c, "stroke-width": 2 }));
    if (sr.points.length <= 30) sr.points.forEach((p) => svg.append(s("circle", { cx: x(p[0]), cy: y(p[1]), r: 3.5, fill: c })));
  });
  if (series.length > 1) legend(svg, series.map((sr, i) => ({ name: sr.name, color: sr.color || PALETTE[i % PALETTE.length] })), w - m.r, m.t);
  return svg;
}

function legend(svg, items, xRight, yTop) {
  items.forEach((it, i) => {
    const y = yTop + 6 + i * 16;
    svg.append(s("rect", { x: xRight - 150, y: y - 6, width: 10, height: 10, rx: 2, fill: it.color }));
    svg.append(s("text", { x: xRight - 135, y: y + 3, class: "tick" }, it.name));
  });
}

// Scatter: points [[x,y]], labels: parallel array of category strings or null
export function scatterChart(points, labels, opts = {}) {
  const w = 560, h = opts.height || 360, m = { l: 56, r: 16, t: 16, b: 44 };
  const svg = frame(w, h, opts.title);
  if (!points.length) return svg;
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const xd = [Math.min(...xs), Math.max(...xs)], yd = [Math.min(...ys), Math.max(...ys)];
  const x = scale(xd, [m.l, w - m.r]), y = scale(yd, [h - m.b, m.t]);
  axes(svg, x, y, xd, yd, m, w, h, opts);
  if (opts.diagonal) svg.append(s("line", { x1: x(xd[0]), y1: y(xd[0]), x2: x(xd[1]), y2: y(xd[1]), class: "diag" }));
  const cats = labels ? [...new Set(labels.map(String))].sort((a, b) => (isNaN(a) || isNaN(b) ? a.localeCompare(b) : a - b)) : [];
  const colorOf = (l) => (l === "-1" || l === -1 ? "#9ca3af" : PALETTE[cats.indexOf(String(l)) % PALETTE.length]);
  points.forEach((p, i) => svg.append(s("circle", { cx: x(p[0]), cy: y(p[1]), r: points.length > 600 ? 2 : 3.2, fill: labels ? colorOf(labels[i]) : PALETTE[0], opacity: 0.75 })));
  if (cats.length && cats.length <= 12) legend(svg, cats.map((c) => ({ name: c === "-1" ? "noise (-1)" : c, color: colorOf(c) })), w - m.r, m.t);
  return svg;
}

// Histogram from {edges, counts}
export function histogram(hist, opts = {}) {
  const mini = !!opts.mini;
  const w = mini ? 300 : 560, h = opts.height || 180, m = mini ? { l: 4, r: 4, t: 4, b: 18 } : { l: 48, r: 10, t: 10, b: 36 };
  const svg = frame(w, h, opts.title);
  if (!hist || !hist.counts || !hist.counts.length) return svg;
  const xd = [hist.edges[0], hist.edges[hist.edges.length - 1]], yd = [0, Math.max(...hist.counts)];
  const x = scale(xd, [m.l, w - m.r]), y = scale(yd, [h - m.b, m.t]);
  if (mini) {
    svg.classList.add("chart-mini"); svg.style.maxWidth = "";
    svg.append(s("text", { x: m.l, y: h - 4, class: "tick" }, nice(xd[0])));
    svg.append(s("text", { x: w - m.r, y: h - 4, "text-anchor": "end", class: "tick" }, nice(xd[1])));
  } else axes(svg, x, y, xd, yd, m, w, h, opts);
  hist.counts.forEach((c, i) => {
    const x0 = x(hist.edges[i]), x1 = x(hist.edges[i + 1]);
    svg.append(s("rect", { x: x0 + 0.5, y: y(c), width: Math.max(x1 - x0 - 1, 1), height: h - m.b - y(c), fill: opts.color || PALETTE[0], opacity: 0.85 }));
  });
  if (opts.zeroLine) svg.append(s("line", { x1: x(0), x2: x(0), y1: m.t, y2: h - m.b, class: "diag" }));
  return svg;
}

// Confusion matrix: square grid with counts shaded by row share.
export function confusionMatrix(matrix, classes) {
  const n = classes.length, cell = Math.max(28, Math.min(64, 420 / n)), labelW = 90;
  const w = labelW + cell * n + 10, h = labelW + cell * n + 10;
  const svg = frame(w, h, "Confusion matrix");
  svg.append(s("text", { x: labelW + (cell * n) / 2, y: 14, "text-anchor": "middle", class: "axis-label" }, "predicted"));
  svg.append(s("text", { x: 12, y: labelW + (cell * n) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${labelW + (cell * n) / 2})`, class: "axis-label" }, "actual"));
  classes.forEach((c, j) => svg.append(s("text", { x: labelW + j * cell + cell / 2, y: labelW - 8, "text-anchor": "middle", class: "tick" }, String(c).slice(0, 10))));
  matrix.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    svg.append(s("text", { x: labelW - 8, y: labelW + i * cell + cell / 2 + 4, "text-anchor": "end", class: "tick" }, String(classes[i]).slice(0, 12)));
    row.forEach((v, j) => {
      const share = v / total;
      svg.append(s("rect", { x: labelW + j * cell, y: labelW + i * cell, width: cell - 1, height: cell - 1, fill: i === j ? "#059669" : "#dc2626", opacity: 0.08 + share * 0.85 }));
      svg.append(s("text", { x: labelW + j * cell + cell / 2, y: labelW + i * cell + cell / 2 + 4, "text-anchor": "middle", class: share > 0.5 ? "cell-light" : "cell" }, v));
    });
  });
  return svg;
}

// Correlation heatmap
export function heatmap(columns, matrix) {
  const n = columns.length, cell = Math.max(16, Math.min(40, 520 / n)), labelW = 110;
  const w = labelW + cell * n + 10, h = labelW + cell * n + 10;
  const svg = frame(w, h, "Correlation matrix");
  columns.forEach((c, j) => svg.append(s("text", { x: labelW + j * cell + cell / 2, y: labelW - 6, "text-anchor": "start", transform: `rotate(-60 ${labelW + j * cell + cell / 2} ${labelW - 6})`, class: "tick" }, String(c).slice(0, 16))));
  matrix.forEach((row, i) => {
    svg.append(s("text", { x: labelW - 6, y: labelW + i * cell + cell / 2 + 4, "text-anchor": "end", class: "tick" }, String(columns[i]).slice(0, 16)));
    row.forEach((v, j) => {
      const fill = v >= 0 ? "#2563eb" : "#dc2626";
      const r = s("rect", { x: labelW + j * cell, y: labelW + i * cell, width: cell - 1, height: cell - 1, fill, opacity: 0.05 + Math.abs(v) * 0.9 });
      r.append(s("title", {}, `${columns[i]} vs ${columns[j]}: ${v.toFixed(2)}`));
      svg.append(r);
      if (cell >= 30) svg.append(s("text", { x: labelW + j * cell + cell / 2, y: labelW + i * cell + cell / 2 + 4, "text-anchor": "middle", class: Math.abs(v) > 0.55 ? "cell-light" : "cell" }, v.toFixed(1)));
    });
  });
  return svg;
}

export { PALETTE };

// Horizontal box plot from {min, q1, median, q3, max, outliers}
export function boxPlot(box, opts = {}) {
  const w = 300, h = 46, m = { l: 6, r: 6 };
  const svg = frame(w, h, opts.title); svg.classList.add("chart-mini"); svg.style.maxWidth = "";
  const vals = [box.min, box.max, ...(box.outliers || [])].filter((v) => v !== null && v !== undefined);
  const xd = [Math.min(...vals), Math.max(...vals)]; if (xd[0] === xd[1]) xd[1] += 1;
  const x = scale(xd, [m.l, w - m.r]); const y = 16;
  svg.append(s("line", { x1: x(box.min), x2: x(box.q1), y1: y, y2: y, stroke: "currentColor", "stroke-width": 1.5, opacity: 0.6 }));
  svg.append(s("line", { x1: x(box.q3), x2: x(box.max), y1: y, y2: y, stroke: "currentColor", "stroke-width": 1.5, opacity: 0.6 }));
  svg.append(s("rect", { x: x(box.q1), y: y - 9, width: Math.max(x(box.q3) - x(box.q1), 1), height: 18, fill: PALETTE[0], opacity: 0.35, rx: 2 }));
  svg.append(s("line", { x1: x(box.median), x2: x(box.median), y1: y - 9, y2: y + 9, stroke: PALETTE[0], "stroke-width": 2 }));
  [box.min, box.max].forEach((v) => svg.append(s("line", { x1: x(v), x2: x(v), y1: y - 5, y2: y + 5, stroke: "currentColor", "stroke-width": 1.5, opacity: 0.6 })));
  (box.outliers || []).forEach((v) => svg.append(s("circle", { cx: x(v), cy: y, r: 2.2, fill: "#dc2626", opacity: 0.8 })));
  svg.append(s("text", { x: m.l, y: h - 4, class: "tick" }, nice(xd[0])));
  svg.append(s("text", { x: w - m.r, y: h - 4, "text-anchor": "end", class: "tick" }, nice(xd[1])));
  return svg;
}
