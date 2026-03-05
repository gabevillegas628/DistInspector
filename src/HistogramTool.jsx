import { useState, useRef, useCallback, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import {
  ComposedChart, Bar, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer
} from "recharts";

// ── CSV parsing ───────────────────────────────────────────────────────────────
function parseCSVRow(line) {
  const fields = [];
  let cur = "";
  let inQuote = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuote && line[i + 1] === '"') { cur += '"'; i++; }
      else inQuote = !inQuote;
    } else if (ch === "," && !inQuote) {
      fields.push(cur.trim());
      cur = "";
    } else {
      cur += ch;
    }
  }
  fields.push(cur.trim());
  return fields;
}

function parseCSV(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim() !== "");
  if (lines.length < 2) return { headers: [], rows: [] };
  const headers = parseCSVRow(lines[0]);
  const rows = lines.slice(1).map(parseCSVRow);
  return { headers, rows };
}

function extractColumn(rows, colIndex) {
  return rows.map(r => parseFloat(r[colIndex])).filter(v => !isNaN(v));
}

// ── Stats helpers ─────────────────────────────────────────────────────────────

function computeStats(arr) {
  const n = arr.length;
  const sorted = [...arr].sort((a, b) => a - b);
  const mean = arr.reduce((s, v) => s + v, 0) / n;
  const median = n % 2 === 0
    ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    : sorted[Math.floor(n / 2)];
  const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const skew = std > 0 ? arr.reduce((s, v) => s + ((v - mean) / std) ** 3, 0) / n : 0;
  const kurt = std > 0 ? arr.reduce((s, v) => s + ((v - mean) / std) ** 4, 0) / n - 3 : 0;
  const q1 = sorted[Math.floor(n * 0.25)];
  const q3 = sorted[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  // Freedman-Diaconis bin width; fall back to Scott's rule if IQR is zero
  const fdBin = iqr > 0
    ? Math.max(0.5, parseFloat((2 * iqr / Math.cbrt(n)).toFixed(2)))
    : Math.max(0.5, parseFloat((3.5 * std / Math.cbrt(n)).toFixed(2)));
  return { n, mean, median, std, skew, kurt, min: sorted[0], max: sorted[n - 1], q1, q3, iqr, fdBin };
}

function computeBins(arr, binSize, stats) {
  if (!binSize || binSize <= 0) return [];
  const start = Math.floor(stats.min / binSize) * binSize;
  const end = Math.ceil(stats.max / binSize) * binSize + binSize;
  const normalPDF = x => Math.exp(-0.5 * ((x - stats.mean) / stats.std) ** 2) / (stats.std * Math.sqrt(2 * Math.PI));
  const bins = [];
  for (let s = start; s < end; s += binSize) {
    const count = arr.filter(v => v >= s && v < s + binSize).length;
    const mid = s + binSize / 2;
    bins.push({
      label: String(parseFloat(s.toFixed(2))),
      range: `${parseFloat(s.toFixed(2))}–${parseFloat((s + binSize).toFixed(2))}`,
      count,
      curve: parseFloat((normalPDF(mid) * arr.length * binSize).toFixed(3)),
    });
  }
  return bins;
}

// ── Local analysis (KDE modality + skewness normality) ────────────────────────

/**
 * Gaussian KDE evaluated over a grid, returns array of density values.
 * Bandwidth via Silverman's rule: h = 1.06 * σ * n^(-1/5)
 */
function computeKDE(data, stats, gridSize = 300) {
  const h = Math.max(1e-9, 1.06 * stats.std * Math.pow(data.length, -0.2));
  const lo = stats.min - 3 * h;
  const hi = stats.max + 3 * h;
  const step = (hi - lo) / gridSize;
  const density = [];
  for (let i = 0; i <= gridSize; i++) {
    const x = lo + i * step;
    const d = data.reduce((sum, xi) =>
      sum + Math.exp(-0.5 * ((x - xi) / h) ** 2), 0
    ) / (data.length * h * Math.sqrt(2 * Math.PI));
    density.push(d);
  }
  return density;
}

/**
 * Counts significant local maxima in the KDE density array.
 * A peak is significant if it exceeds `threshold` fraction of the global max.
 * Adjacent peaks must be separated by a valley that drops below peakSep * peak height.
 */
function countKDEPeaks(density, threshold = 0.12, peakSep = 0.6) {
  const maxD = Math.max(...density);
  if (maxD === 0) return 1;

  // Find all local maxima above threshold
  const rawPeaks = [];
  for (let i = 1; i < density.length - 1; i++) {
    if (density[i] > density[i - 1] && density[i] > density[i + 1] && density[i] >= maxD * threshold) {
      rawPeaks.push({ idx: i, val: density[i] });
    }
  }

  if (rawPeaks.length <= 1) return rawPeaks.length || 1;

  // Merge peaks that aren't separated by a deep enough valley
  const significant = [rawPeaks[0]];
  for (let k = 1; k < rawPeaks.length; k++) {
    const prev = significant[significant.length - 1];
    const curr = rawPeaks[k];
    const valleyMin = Math.min(...density.slice(prev.idx, curr.idx + 1));
    const minPeak = Math.min(prev.val, curr.val);
    if (valleyMin < minPeak * peakSep) {
      significant.push(curr);
    } else {
      // Absorb into the taller peak
      if (curr.val > prev.val) significant[significant.length - 1] = curr;
    }
  }
  return significant.length;
}

function detectModality(data, stats) {
  const density = computeKDE(data, stats);
  const peaks = countKDEPeaks(density);

  if (peaks <= 1) return { label: "unimodal", note: "A single dominant peak was detected via kernel density estimation." };
  if (peaks === 2) return { label: "likely bimodal", note: "Two distinct peaks were detected via kernel density estimation, suggesting two subgroups or processes." };
  return { label: "likely multimodal", note: `${peaks} peaks detected via kernel density estimation, suggesting multiple subgroups or underlying processes.` };
}

function assessNormality(stats) {
  const s = stats.skew;
  const k = stats.kurt; // excess kurtosis
  if (Math.abs(s) < 0.5 && Math.abs(k) < 1) return "approximately normal";
  if (k > 3) return "heavy-tailed";
  if (s >= 1.0) return "heavy positive skew";
  if (s <= -1.0) return "heavy negative skew";
  if (s >= 0.5) return "slight positive skew";
  if (s <= -0.5) return "slight negative skew";
  return "approximately normal";
}

function generateInterpretation(stats, modality, normalityAssessment) {
  const parts = [];

  parts.push(
    `The distribution spans ${stats.min} to ${stats.max} with a mean of ${stats.mean.toFixed(2)} and standard deviation of ${stats.std.toFixed(2)}.`
  );

  if (modality.label !== "unimodal") {
    parts.push(
      `The shape appears ${modality.label}, which may indicate multiple subgroups or distinct processes within the data.`
    );
  } else if (normalityAssessment === "approximately normal") {
    parts.push("The distribution is roughly symmetric and bell-shaped around the mean.");
  } else if (normalityAssessment === "heavy-tailed") {
    parts.push("The distribution has heavier tails than a normal distribution, with more extreme values than expected.");
  } else {
    const dir = stats.skew > 0 ? "right (positive)" : "left (negative)";
    parts.push(`The distribution is skewed ${dir}, meaning the tail extends ${stats.skew > 0 ? "toward higher" : "toward lower"} values.`);
  }

  const meanMedianGap = Math.abs(stats.mean - stats.median);
  if (stats.std > 0 && meanMedianGap / stats.std > 0.15) {
    const dir = stats.mean > stats.median ? "above" : "below";
    parts.push(
      `The mean (${stats.mean.toFixed(2)}) sits ${dir} the median (${stats.median.toFixed(2)}), consistent with the observed skew.`
    );
  }

  return parts.join(" ");
}

function runLocalAnalysis(data, stats) {
  const modality = detectModality(data, stats);
  const suggestedBinWidth = stats.fdBin;
  const normalityAssessment = assessNormality(stats);
  const interpretation = generateInterpretation(stats, modality, normalityAssessment);
  return {
    modality: modality.label,
    modalityNote: modality.note,
    suggestedBinWidth,
    normalityAssessment,
    interpretation,
  };
}

// ── PNG Export ────────────────────────────────────────────────────────────────
const EXPORT_SIZES = [
  { id: "large-rect",  label: "Large Rectangle", w: 1600, h: 900,  pvW: 60, pvH: 34, modalW: 200, modalH: 112 },
  { id: "small-rect",  label: "Small Rectangle",  w: 800,  h: 450,  pvW: 44, pvH: 25, modalW: 200, modalH: 112 },
  { id: "large-sq",    label: "Large Square",     w: 1200, h: 1200, pvW: 38, pvH: 38, modalW: 140, modalH: 140 },
  { id: "small-sq",    label: "Small Square",     w: 600,  h: 600,  pvW: 28, pvH: 28, modalW: 140, modalH: 140 },
];

const EXPORT_STAT_OPTIONS = [
  { key: "n",        label: "n",        color: "#818cf8" },
  { key: "mean",     label: "Mean",     color: "#38bdf8" },
  { key: "median",   label: "Median",   color: "#34d399" },
  { key: "sd",       label: "SD",       color: "#fbbf24" },
  { key: "skew",     label: "Skew",     color: "#f472b6" },
  { key: "min",      label: "Min",      color: "#64748b" },
  { key: "max",      label: "Max",      color: "#64748b" },
  { key: "modality", label: "Modality", color: "#a78bfa" },
  { key: "shape",    label: "Shape",    color: "#94a3b8" },
];

// ── Export colour themes ──────────────────────────────────────────────────────
const EXPORT_DARK  = { bg: "#0a0f1e", surface: "#0f172a", grid: "#1e293b", tick: "#475569", title: "#f1f5f9", pillBg: "#0f172a", pillLbl: "#475569" };
const EXPORT_LIGHT = { bg: "#f8fafc", surface: "#ffffff", grid: "#e2e8f0", tick: "#64748b", title: "#0f172a", pillBg: "#ffffff", pillLbl: "#64748b" };

// ── Off-screen chart renderer (fixed size, window-independent) ────────────────
function OffscreenChart({ width, height, bins, stats, binSize, et }) {
  const meanBinX = String(parseFloat(
    (Math.floor((stats.mean - Math.floor(stats.min / binSize) * binSize) / binSize) * binSize
      + Math.floor(stats.min / binSize) * binSize).toFixed(2)
  ));
  return (
    <ComposedChart data={bins} width={width} height={height} margin={{ top: 28, right: 16, left: 0, bottom: 22 }}>
      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={et.grid} />
      <XAxis dataKey="label" tick={{ fontSize: 12, fill: et.tick }} interval="preserveStartEnd" />
      <YAxis tick={{ fontSize: 12, fill: et.tick }} label={{ value: "Count", angle: -90, position: "insideLeft", offset: 10, fontSize: 13, fill: et.tick }} allowDecimals={false} />
      <ReferenceLine x={meanBinX} stroke="#38bdf8" strokeDasharray="4 3" strokeWidth={1.5}
        label={<MeanPillLabel value={`μ=${stats.mean.toFixed(1)}`} pillBg={et.surface} />} />
      <Bar dataKey="count" fill="#6366f1" fillOpacity={0.8} radius={[3, 3, 0, 0]} isAnimationActive={false} />
      <Line dataKey="curve" type="monotone" stroke="#fbbf24" strokeWidth={2} dot={false} isAnimationActive={false} />
    </ComposedChart>
  );
}

async function exportPNG({ titleText, sizeId, statKeys, stats, analysis, bins, binSize, dark = true }) {
  const et = dark ? EXPORT_DARK : EXPORT_LIGHT;
  const size = EXPORT_SIZES.find(s => s.id === sizeId) ?? EXPORT_SIZES[0];
  const W = size.w, H = size.h;

  const pad       = Math.round(Math.max(24, W * 0.025));
  const titleSize = Math.round(Math.max(14, W * 0.02));
  const titleAreaH = titleSize + pad;

  const statValues = {
    n:        String(stats.n),
    mean:     stats.mean.toFixed(2),
    median:   stats.median.toFixed(2),
    sd:       stats.std.toFixed(2),
    skew:     stats.skew.toFixed(3),
    min:      String(stats.min),
    max:      String(stats.max),
    modality: analysis?.modality ?? "",
    shape:    analysis?.normalityAssessment ?? "",
  };
  const statColors = {
    n: "#818cf8", mean: "#38bdf8", median: "#34d399", sd: "#fbbf24",
    skew: "#f472b6", min: "#64748b", max: "#64748b",
    modality: MODALITY_COLOR[analysis?.modality] ?? "#94a3b8",
    shape: "#94a3b8",
  };
  const legendItems = statKeys
    .filter(k => statValues[k])
    .map(k => ({ label: EXPORT_STAT_OPTIONS.find(o => o.key === k)?.label ?? k, value: statValues[k], color: statColors[k] }));

  const pillH  = Math.round(Math.min(H * 0.065, 54));
  const legendH = legendItems.length > 0 ? pillH + Math.round(pad * 0.6) : 0;
  const legendGap = legendH > 0 ? Math.round(pad * 0.35) : 0;
  const legendBottomPad = legendH > 0 ? Math.round(pad * 0.5) : 0;
  const chartY  = titleAreaH;
  const chartH  = H - chartY - pad - legendGap - legendH - legendBottomPad;
  const chartW  = W - 2 * pad;

  const canvas = document.createElement("canvas");
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = et.bg;
  ctx.fillRect(0, 0, W, H);

  ctx.fillStyle = et.title;
  ctx.font = `bold ${titleSize}px system-ui, -apple-system, sans-serif`;
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(titleText || "Distribution", pad, Math.round(pad * 0.6));

  // Render chart at exact export dimensions, independent of browser window size
  const offscreen = document.createElement("div");
  offscreen.style.cssText = "position:fixed;left:-9999px;top:-9999px;pointer-events:none;";
  document.body.appendChild(offscreen);
  const offRoot = createRoot(offscreen);
  flushSync(() => {
    offRoot.render(<OffscreenChart width={chartW} height={chartH} bins={bins} stats={stats} binSize={binSize} et={et} />);
  });
  const svgEl = offscreen.querySelector("svg");
  const xml = svgEl ? new XMLSerializer().serializeToString(svgEl) : null;
  offRoot.unmount();
  document.body.removeChild(offscreen);
  if (!xml) return;

  const url = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml;charset=utf-8" }));
  await new Promise(resolve => {
    const img = new Image();
    img.onload = () => { ctx.drawImage(img, pad, chartY, chartW, chartH); URL.revokeObjectURL(url); resolve(); };
    img.onerror = resolve;
    img.src = url;
  });

  // Legend pills
  if (legendItems.length > 0) {
    const lFontVal = Math.min(Math.round(pillH * 0.38), 16);
    const lFontLbl = Math.min(Math.round(pillH * 0.26), 10);
    const pillPadX = Math.round(pillH * 0.4);
    const gap      = Math.round(W * 0.008);
    const legendY  = chartY + chartH + legendGap;
    let cx = pad;

    ctx.textBaseline = "alphabetic";
    for (const { label, value, color } of legendItems) {
      ctx.font = `bold ${lFontVal}px 'DM Mono', monospace`;
      const vW = ctx.measureText(value).width;
      ctx.font = `${lFontLbl}px system-ui`;
      const lW = ctx.measureText(label.toUpperCase()).width;
      const pw = Math.max(vW, lW) + 2 * pillPadX;
      if (cx + pw > W - pad) break;

      ctx.fillStyle = et.pillBg;
      ctx.strokeStyle = color + "44";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(cx, legendY, pw, pillH, 6);
      ctx.fill(); ctx.stroke();

      ctx.textAlign = "center";
      ctx.fillStyle = et.pillLbl;
      ctx.font = `${lFontLbl}px system-ui`;
      ctx.fillText(label.toUpperCase(), cx + pw / 2, legendY + lFontLbl + Math.round(pillH * 0.12));

      ctx.fillStyle = color;
      ctx.font = `bold ${lFontVal}px 'DM Mono', monospace`;
      ctx.fillText(value, cx + pw / 2, legendY + pillH - Math.round(pillH * 0.15));

      cx += pw + gap;
    }
  }

  canvas.toBlob(blob => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${(titleText || "histogram").replace(/\s+/g, "_")}.png`;
    a.click();
  }, "image/png");
}

// ── Modal chart preview (live data, correct aspect ratio) ─────────────────────
function ModalChartPreview({ previewW, previewH, bins, stats, binSize, active, et }) {
  const meanBinX = String(parseFloat(
    (Math.floor((stats.mean - Math.floor(stats.min / binSize) * binSize) / binSize) * binSize
      + Math.floor(stats.min / binSize) * binSize).toFixed(2)
  ));
  return (
    <ComposedChart data={bins} width={previewW} height={previewH} margin={{ top: 6, right: 4, left: 4, bottom: 4 }}>
      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={et.grid} />
      <XAxis dataKey="label" hide />
      <YAxis hide allowDecimals={false} />
      <ReferenceLine x={meanBinX} stroke="#38bdf8" strokeDasharray="3 2" strokeWidth={1} />
      <Bar dataKey="count" fill="#6366f1" fillOpacity={active ? 0.85 : 0.5} radius={[2, 2, 0, 0]} isAnimationActive={false} />
      <Line dataKey="curve" type="monotone" stroke="#fbbf24" strokeWidth={1.5} dot={false} isAnimationActive={false} />
    </ComposedChart>
  );
}

// ── Export modal ──────────────────────────────────────────────────────────────
function ExportModal({ open, onClose, onExport, bins, stats, binSize }) {
  const [sizeId, setSizeId] = useState("large-rect");
  const [statKeys, setStatKeys] = useState(EXPORT_STAT_OPTIONS.map(o => o.key));
  const [exportDark, setExportDark] = useState(true);
  const et = exportDark ? EXPORT_DARK : EXPORT_LIGHT;
  if (!open) return null;

  const toggleStat = key =>
    setStatKeys(prev => prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]);
  const allSelected = statKeys.length === EXPORT_STAT_OPTIONS.length;

  return (
    <div
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
      style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.65)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
    >
      <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 14, padding: 28, width: 520, maxWidth: "92vw", maxHeight: "90vh", overflowY: "auto" }}>

        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
          <span style={{ fontSize: 15, fontWeight: 700, color: "#f1f5f9" }}>Export PNG</span>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ display: "flex", background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 6, overflow: "hidden" }}>
              {[["Dark", true], ["Light", false]].map(([label, val]) => (
                <button key={label} onClick={() => setExportDark(val)} style={{
                  background: exportDark === val ? "#1e293b" : "transparent",
                  border: "none", color: exportDark === val ? "#e2e8f0" : "#475569",
                  fontSize: 11, fontWeight: 600, padding: "5px 12px", cursor: "pointer",
                  fontFamily: "inherit", transition: "all 0.12s",
                }}>{label}</button>
              ))}
            </div>
            <button onClick={onClose} style={{ background: "none", border: "none", color: "#475569", cursor: "pointer", fontSize: 22, lineHeight: 1, padding: "0 2px" }}>×</button>
          </div>
        </div>

        {/* Size selector */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: "0.14em", marginBottom: 10 }}>Size</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 10 }}>
            {EXPORT_SIZES.map(s => {
              const active = sizeId === s.id;
              return (
                <div key={s.id} onClick={() => setSizeId(s.id)} style={{
                  background: active ? "#1e293b" : "#0a0f1e",
                  border: `1.5px solid ${active ? "#6366f1" : "#1e293b"}`,
                  borderRadius: 8, padding: "10px 10px 8px", cursor: "pointer",
                  display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
                  transition: "border-color 0.15s",
                }}>
                  <div style={{ borderRadius: 4, overflow: "hidden", border: `1px solid ${active ? "#334155" : "#1a2a3a"}`, lineHeight: 0, background: et.bg }}>
                    {bins && stats ? (
                      <ModalChartPreview previewW={s.modalW} previewH={s.modalH} bins={bins} stats={stats} binSize={binSize} active={active} et={et} />
                    ) : (
                      <div style={{ width: s.modalW, height: s.modalH, background: et.bg }} />
                    )}
                  </div>
                  <div style={{ fontSize: 10, fontWeight: 600, color: active ? "#e2e8f0" : "#64748b", textAlign: "center", lineHeight: 1.3 }}>{s.label}</div>
                  <div style={{ fontSize: 9, color: "#475569", fontFamily: "'DM Mono', monospace" }}>{s.w}×{s.h}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Legend stats */}
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
            <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: "0.14em" }}>Legend stats</div>
            <button
              onClick={() => setStatKeys(allSelected ? [] : EXPORT_STAT_OPTIONS.map(o => o.key))}
              style={{ fontSize: 10, color: "#475569", background: "none", border: "none", cursor: "pointer", padding: 0, fontFamily: "inherit", textDecoration: "underline" }}
            >
              {allSelected ? "deselect all" : "select all"}
            </button>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {EXPORT_STAT_OPTIONS.map(opt => {
              const active = statKeys.includes(opt.key);
              return (
                <div key={opt.key} onClick={() => toggleStat(opt.key)} style={{
                  background: active ? "#1e293b" : "#0a0f1e",
                  border: `1px solid ${active ? opt.color + "55" : "#1e293b"}`,
                  borderRadius: 6, padding: "5px 12px", cursor: "pointer",
                  display: "flex", alignItems: "center", gap: 6, transition: "all 0.12s",
                }}>
                  <div style={{ width: 7, height: 7, borderRadius: 2, background: active ? opt.color : "#334155", flexShrink: 0, transition: "background 0.12s" }} />
                  <span style={{ fontSize: 11, color: active ? "#e2e8f0" : "#475569", fontWeight: 500 }}>{opt.label}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Export button */}
        <button
          onClick={() => onExport({ sizeId, statKeys, dark: exportDark })}
          style={{ width: "100%", background: "linear-gradient(135deg, #6366f1 0%, #818cf8 100%)", border: "none", borderRadius: 8, color: "white", fontFamily: "inherit", fontWeight: 600, fontSize: 14, padding: "13px", cursor: "pointer" }}
        >
          ↓ Export PNG
        </button>
      </div>
    </div>
  );
}

// ── Stat pill with tooltip ────────────────────────────────────────────────────
function StatPill({ label, value, color, tip, small }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      style={{ position: "relative" }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {hovered && (
        <div style={{
          position: "absolute", bottom: "calc(100% + 7px)", left: "50%",
          transform: "translateX(-50%)", background: "#1e293b",
          border: "1px solid #334155", borderRadius: 6, padding: "7px 11px",
          fontSize: 11, color: "#94a3b8", zIndex: 20, pointerEvents: "none",
          width: 190, textAlign: "center", lineHeight: 1.55,
          boxShadow: "0 4px 16px #00000066",
        }}>
          {tip}
          <div style={{
            position: "absolute", top: "100%", left: "50%", transform: "translateX(-50%)",
            borderLeft: "5px solid transparent", borderRight: "5px solid transparent",
            borderTop: "5px solid #334155",
          }} />
        </div>
      )}
      <div style={{ background: "#0f172a", border: `1px solid ${color}30`, borderRadius: 6, padding: "5px 13px", display: "flex", flexDirection: "column", alignItems: "center", cursor: "default" }}>
        <span style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: "0.1em" }}>{label}</span>
        <span style={{ fontSize: small ? 12 : 14, fontWeight: 700, color, fontFamily: "'DM Mono', monospace" }}>{value}</span>
      </div>
    </div>
  );
}

// ── Mean reference line pill label ────────────────────────────────────────────
function MeanPillLabel({ viewBox, value, pillBg }) {
  if (!viewBox) return null;
  const { x, y } = viewBox;
  const pw = 68, ph = 20, px = x - pw / 2, py = y - ph - 6;
  return (
    <g>
      <rect x={px} y={py} width={pw} height={ph} rx={5} ry={5}
        fill={pillBg ?? "#0f172a"} stroke="#38bdf8" strokeWidth={1.2} />
      <text x={x} y={py + ph / 2 + 4.5} textAnchor="middle"
        fill="#38bdf8" fontSize={12} fontWeight={700} fontFamily="'DM Mono', monospace">
        {value}
      </text>
    </g>
  );
}

// ── Modal badge colors ─────────────────────────────────────────────────────────
const MODALITY_COLOR = {
  "unimodal": "#34d399",
  "likely bimodal": "#fbbf24",
  "likely multimodal": "#f87171",
  "uncertain": "#94a3b8",
};

// ── Custom tooltip ─────────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, n }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, padding: "8px 14px", fontSize: 12, color: "#e2e8f0" }}>
      <div style={{ fontWeight: 700, marginBottom: 2 }}>{d.range}</div>
      <div style={{ color: "#a78bfa" }}>{d.count} values ({n ? (d.count / n * 100).toFixed(1) : 0}%)</div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function HistogramTool() {
  const [csvData, setCsvData] = useState(null);      // { headers, rows, fileName }
  const [selectedCol, setSelectedCol] = useState(""); // column index as string
  const [scores, setScores] = useState(null);
  const [stats, setStats] = useState(null);
  const [binSize, setBinSize] = useState(8);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState("");
  const [title, setTitle] = useState("Distribution");
  const [titleFocused, setTitleFocused] = useState(false);
  const [exportModalOpen, setExportModalOpen] = useState(false);
  const [filterOutliers, setFilterOutliers] = useState(false);
  const [filteredStats, setFilteredStats] = useState(null);
  const [filteredAnalysis, setFilteredAnalysis] = useState(null);
  const chartRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!filterOutliers || !scores || !stats || stats.iqr === 0) {
      setFilteredStats(null);
      setFilteredAnalysis(null);
      return;
    }
    const lo = stats.q1 - 1.5 * stats.iqr;
    const hi = stats.q3 + 1.5 * stats.iqr;
    const filtered = scores.filter(v => v >= lo && v <= hi);
    if (filtered.length < 5) return;
    const fs = computeStats(filtered);
    setFilteredStats(fs);
    setFilteredAnalysis(runLocalAnalysis(filtered, fs));
  }, [filterOutliers, scores, stats]);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!file.name.endsWith(".csv")) { setError("Please upload a .csv file."); return; }
    const reader = new FileReader();
    reader.onload = (e) => {
      const { headers, rows } = parseCSV(e.target.result);
      if (headers.length === 0) { setError("Could not parse CSV — check the file format."); return; }
      setCsvData({ headers, rows, fileName: file.name });
      setSelectedCol("");
      setScores(null);
      setStats(null);
      setAnalysis(null);
      setError("");
      setTitle(file.name.replace(/\.csv$/i, ""));
    };
    reader.readAsText(file);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const handleColChange = useCallback((colIdx) => {
    setSelectedCol(colIdx);
    if (colIdx === "" || !csvData) return;
    const parsed = extractColumn(csvData.rows, parseInt(colIdx, 10));
    if (parsed.length < 5) { setError("This column has fewer than 5 numeric values."); return; }
    setError("");
    const s = computeStats(parsed);
    const result = runLocalAnalysis(parsed, s);
    setScores(parsed);
    setStats(s);
    setBinSize(result.suggestedBinWidth);
    setAnalysis(result);
    setFilterOutliers(false);
    setFilteredStats(null);
    setFilteredAnalysis(null);
  }, [csvData]);

  const lo = (stats && stats.iqr > 0) ? stats.q1 - 1.5 * stats.iqr : -Infinity;
  const hi = (stats && stats.iqr > 0) ? stats.q3 + 1.5 * stats.iqr : Infinity;
  const activeScores = (scores && filterOutliers) ? scores.filter(v => v >= lo && v <= hi) : scores;
  const outlierCount = (scores && filterOutliers) ? scores.length - (activeScores?.length ?? 0) : 0;
  const displayStats = (filterOutliers && filteredStats) ? filteredStats : stats;
  const displayAnalysis = (filterOutliers && filteredAnalysis) ? filteredAnalysis : analysis;
  const bins = activeScores && displayStats ? computeBins(activeScores, binSize, displayStats) : [];
  const maxBin = displayStats ? Math.max(5, Math.round((displayStats.max - displayStats.min) / 3)) : 20;

  return (
    <div style={{ minHeight: "100vh", background: "#0a0f1e", color: "#e2e8f0", fontFamily: "'DM Sans', system-ui, sans-serif", padding: "28px 24px", boxSizing: "border-box" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#818cf8", boxShadow: "0 0 8px #818cf899" }} />
          <span style={{ fontSize: 10, letterSpacing: "0.18em", color: "#475569", textTransform: "uppercase" }}>Distribution Inspector</span>
          <span style={{ fontSize: 10, color: "#1e293b", marginLeft: "auto" }}>Title included in PNG export</span>
        </div>
        <div style={{ borderBottom: `1px solid ${titleFocused ? "#6366f1" : "#1e293b"}`, transition: "border-color 0.15s", paddingBottom: 4 }}>
          <input
            value={title}
            onChange={e => setTitle(e.target.value)}
            onFocus={() => setTitleFocused(true)}
            onBlur={() => setTitleFocused(false)}
            placeholder="Dataset title…"
            style={{ fontSize: 22, fontWeight: 700, background: "transparent", border: "none", color: "#f1f5f9", outline: "none", width: "100%", fontFamily: "inherit", padding: 0 }}
          />
        </div>
      </div>

      {/* CSV upload + column selector */}
      <div style={{ display: "flex", gap: 10, marginBottom: 12, alignItems: "stretch", flexWrap: "wrap" }}>
        {/* Drop zone */}
        <div
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
          style={{
            flex: "1 1 260px", minHeight: 72, background: "#0f172a",
            border: `2px dashed ${csvData ? "#6366f1" : "#1e293b"}`,
            borderRadius: 8, display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", gap: 6,
            cursor: "pointer", transition: "border-color 0.15s", padding: "12px 20px",
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            style={{ display: "none" }}
            onChange={e => handleFile(e.target.files[0])}
          />
          {csvData ? (
            <>
              <span style={{ fontSize: 18 }}>📄</span>
              <span style={{ fontSize: 12, color: "#818cf8", fontWeight: 600 }}>{csvData.fileName}</span>
              <span style={{ fontSize: 10, color: "#475569" }}>{csvData.rows.length} rows · {csvData.headers.length} columns — click to replace</span>
            </>
          ) : (
            <>
              <span style={{ fontSize: 18, opacity: 0.4 }}>⬆</span>
              <span style={{ fontSize: 12, color: "#334155" }}>Drop a CSV here or click to browse</span>
            </>
          )}
        </div>

        {/* Column selector */}
        {csvData && (
          <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", gap: 6, flex: "0 0 200px" }}>
            <span style={{ fontSize: 10, color: "#475569", textTransform: "uppercase", letterSpacing: "0.12em" }}>Select column</span>
            <select
              value={selectedCol}
              onChange={e => handleColChange(e.target.value)}
              style={{
                background: "#0f172a", border: "1px solid #334155", borderRadius: 6,
                color: selectedCol === "" ? "#475569" : "#e2e8f0",
                fontFamily: "inherit", fontSize: 13, padding: "8px 12px", outline: "none", cursor: "pointer",
              }}
            >
              <option value="">— choose a column —</option>
              {csvData.headers.map((h, i) => (
                <option key={i} value={i}>{h || `Column ${i + 1}`}</option>
              ))}
            </select>
          </div>
        )}
      </div>
      {error && <div style={{ color: "#f87171", fontSize: 12, marginBottom: 10 }}>{error}</div>}

      {/* Stat pills */}
      {displayStats && (
        <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
          <StatPill label="n" value={displayStats.n} color="#818cf8" tip="Sample size — the total number of numeric values in the selected column." />
          <StatPill label="mean" value={displayStats.mean.toFixed(1)} color="#38bdf8" tip="Arithmetic average. Sum of all values divided by n. Sensitive to outliers." />
          <StatPill label="median" value={displayStats.median.toFixed(1)} color="#34d399" tip="Middle value when data is sorted. More robust than the mean when outliers are present." />
          <StatPill label="SD" value={displayStats.std.toFixed(1)} color="#fbbf24" tip="Standard deviation — the typical distance of values from the mean. Larger = more spread out." />
          <StatPill label="skew" value={displayStats.skew.toFixed(2)} color="#f472b6" tip="Skewness measures asymmetry. 0 = symmetric. Positive = long right tail (a few very high values). Negative = long left tail (a few very low values)." />
          <StatPill label="min" value={displayStats.min} color="#64748b" tip="Smallest value in the dataset." />
          <StatPill label="max" value={displayStats.max} color="#64748b" tip="Largest value in the dataset." />
          {displayAnalysis && (
            <StatPill
              label="modality"
              value={displayAnalysis.modality}
              color={MODALITY_COLOR[displayAnalysis.modality] || "#94a3b8"}
              tip="Number of peaks detected via kernel density estimation. Bimodal or multimodal distributions often indicate distinct subgroups in the data."
              small
            />
          )}
        </div>
      )}

      {/* Chart + Analysis */}
      {scores && displayStats && (
        <div style={{ display: "flex", gap: 16, marginBottom: 16, flexWrap: "wrap" }}>
          {/* Chart */}
          <div ref={chartRef} style={{ flex: "1 1 400px", background: "#0f172a", borderRadius: 10, padding: "16px 8px 8px", border: "1px solid #1e293b", minWidth: 0 }}>
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={bins} margin={{ top: 28, right: 16, left: 0, bottom: 22 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                <XAxis dataKey="label" tick={{ fontSize: 12, fill: "#475569" }} interval="preserveStartEnd" />
                <YAxis tick={{ fontSize: 12, fill: "#475569" }} label={{ value: "Count", angle: -90, position: "insideLeft", offset: 10, fontSize: 13, fill: "#475569" }} allowDecimals={false} />
                <Tooltip content={<ChartTooltip n={displayStats.n} />} cursor={{ fill: "#1e293b" }} />
                <ReferenceLine
                  x={String(parseFloat((Math.floor((displayStats.mean - Math.floor(displayStats.min / binSize) * binSize) / binSize) * binSize + Math.floor(displayStats.min / binSize) * binSize).toFixed(2)))}
                  stroke="#38bdf8" strokeDasharray="4 3" strokeWidth={1.5}
                  label={<MeanPillLabel value={`μ=${displayStats.mean.toFixed(1)}`} />}
                />
                <Bar dataKey="count" fill="#6366f1" fillOpacity={0.8} radius={[3, 3, 0, 0]} />
                <Line dataKey="curve" type="monotone" stroke="#fbbf24" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{ display: "flex", gap: 16, justifyContent: "center", fontSize: 10, color: "#475569", marginTop: 4 }}>
              <span style={{ display: "flex", alignItems: "center", gap: 5 }}><span style={{ display: "inline-block", width: 10, height: 10, background: "#6366f1", borderRadius: 2 }} /> Counts</span>
              <span style={{ display: "flex", alignItems: "center", gap: 5 }}><span style={{ display: "inline-block", width: 16, height: 2, background: "#fbbf24" }} /> Normal fit</span>
              <span style={{ display: "flex", alignItems: "center", gap: 5 }}><span style={{ display: "inline-block", width: 16, height: 2, background: "#38bdf8", borderTop: "1px dashed #38bdf8" }} /> Mean</span>
            </div>
          </div>

          {/* Analysis panel */}
          {displayAnalysis && (
            <div style={{ flex: "0 0 220px", background: "#0f172a", borderRadius: 10, padding: "18px", border: "1px solid #1e293b", display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <div style={{ fontSize: 9, color: "#475569", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: 5 }}>Modality</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: MODALITY_COLOR[displayAnalysis.modality] || "#e2e8f0", marginBottom: 5 }}>{displayAnalysis.modality}</div>
                <p style={{ fontSize: 11, color: "#64748b", margin: 0, lineHeight: 1.6 }}>{displayAnalysis.modalityNote}</p>
              </div>
              <div style={{ borderTop: "1px solid #1e293b", paddingTop: 14 }}>
                <div style={{ fontSize: 9, color: "#475569", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: 5 }}>Shape</div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#e2e8f0" }}>{displayAnalysis.normalityAssessment}</div>
              </div>
              <div style={{ borderTop: "1px solid #1e293b", paddingTop: 14, flex: 1 }}>
                <div style={{ fontSize: 9, color: "#475569", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: 5 }}>Interpretation</div>
                <p style={{ fontSize: 11, color: "#94a3b8", margin: 0, lineHeight: 1.7 }}>{displayAnalysis.interpretation}</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Controls bar */}
      {scores && displayStats && (
        <div style={{ display: "flex", alignItems: "center", gap: 16, background: "#0f172a", borderRadius: 8, padding: "12px 18px", border: "1px solid #1e293b", flexWrap: "wrap" }}>
          <span style={{ fontSize: 11, color: "#475569", whiteSpace: "nowrap" }}>Bin width</span>
          <input type="range" min={0.5} max={maxBin} step={0.5} value={binSize} onChange={e => setBinSize(parseFloat(e.target.value))} style={{ flex: 1, minWidth: 80, accentColor: "#6366f1" }} />
          <span style={{ fontSize: 13, fontFamily: "'DM Mono', monospace", color: "#818cf8", minWidth: 28 }}>{binSize}</span>
          <button
            onClick={() => setBinSize(displayStats.fdBin)}
            style={{ fontSize: 11, background: "#1e293b", border: "1px solid #334155", borderRadius: 4, color: "#64748b", padding: "4px 10px", cursor: "pointer", fontFamily: "inherit" }}
          >
            Auto
          </button>
          <div style={{ width: 1, height: 20, background: "#1e293b" }} />
          <button
            onClick={() => setFilterOutliers(f => !f)}
            style={{
              background: filterOutliers ? "#1e1030" : "#1e293b",
              border: `1px solid ${filterOutliers ? "#a78bfa55" : "#334155"}`,
              borderRadius: 6, color: filterOutliers ? "#a78bfa" : "#64748b",
              fontFamily: "inherit", fontWeight: 500, fontSize: 12,
              padding: "7px 14px", cursor: "pointer", display: "flex", alignItems: "center", gap: 6,
              whiteSpace: "nowrap", transition: "all 0.15s",
            }}
          >
            Filter outliers
            {filterOutliers && outlierCount > 0 && (
              <span style={{ fontSize: 10, background: "#a78bfa22", borderRadius: 10, padding: "1px 6px", color: "#a78bfa", fontFamily: "'DM Mono', monospace" }}>
                -{outlierCount}
              </span>
            )}
          </button>
          <div style={{ width: 1, height: 20, background: "#1e293b" }} />
          <button
            onClick={() => setExportModalOpen(true)}
            style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, color: "#e2e8f0", fontFamily: "inherit", fontWeight: 500, fontSize: 12, padding: "7px 14px", cursor: "pointer", display: "flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}
          >
            ↓ Export PNG
          </button>
        </div>
      )}

      <ExportModal
        open={exportModalOpen}
        onClose={() => setExportModalOpen(false)}
        bins={bins}
        stats={displayStats}
        binSize={binSize}
        onExport={({ sizeId, statKeys, dark }) => {
          setExportModalOpen(false);
          setTimeout(() => exportPNG({ titleText: title, sizeId, statKeys, stats: displayStats, analysis: displayAnalysis, bins, binSize, dark }), 0);
        }}
      />

      {/* Empty state */}
      {!scores && (
        <div style={{ textAlign: "center", padding: "60px 0" }}>
          <div style={{ fontSize: 48, marginBottom: 12, opacity: 0.15, color: "#818cf8" }}>▦</div>
          {!csvData
            ? <div style={{ fontSize: 13, color: "#334155" }}>Upload a CSV to get started</div>
            : <div style={{ fontSize: 13, color: "#334155" }}>Select a numeric column above</div>
          }
        </div>
      )}
    </div>
  );
}
