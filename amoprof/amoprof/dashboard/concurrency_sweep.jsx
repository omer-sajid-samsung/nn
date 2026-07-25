import { useState, useCallback } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, AreaChart
} from "recharts";

// ── Demo data (replace with paste from concurrency_sweep.csv) ─────────────────
const DEMO_DATA = [
  { inference_concurrency:  1, ttft_mean_ms:  4820, ttft_p99_ms:  9200, tpot_mean_ms:  328, tpot_p99_ms:  720, throughput_req_s: 0.01, throughput_tok_s:   3, prompt_tokens_mean: 4800, output_tokens_mean:  310, hbm_used_gb_mean:  72, hbm_util_pct_mean:  22, dram_used_gb_mean: 48, dram_util_pct_mean:  9, ssd_read_bw_mb_mean:  0.0, ssd_write_bw_mb_mean:  0.0, ssd_read_iops_mean:    0, ssd_util_pct_mean:  0.1, kv_cache_hit_rate_pct:  0 },
  { inference_concurrency:  4, ttft_mean_ms:  5100, ttft_p99_ms: 11400, tpot_mean_ms:  342, tpot_p99_ms:  890, throughput_req_s: 0.04, throughput_tok_s:  12, prompt_tokens_mean: 4820, output_tokens_mean:  305, hbm_used_gb_mean:  95, hbm_util_pct_mean:  30, dram_used_gb_mean: 49, dram_util_pct_mean: 10, ssd_read_bw_mb_mean:  0.0, ssd_write_bw_mb_mean:  0.0, ssd_read_iops_mean:    0, ssd_util_pct_mean:  0.1, kv_cache_hit_rate_pct:  0 },
  { inference_concurrency:  8, ttft_mean_ms:  6200, ttft_p99_ms: 15800, tpot_mean_ms:  378, tpot_p99_ms: 1100, throughput_req_s: 0.09, throughput_tok_s:  26, prompt_tokens_mean: 4790, output_tokens_mean:  298, hbm_used_gb_mean: 148, hbm_util_pct_mean:  46, dram_used_gb_mean: 50, dram_util_pct_mean: 10, ssd_read_bw_mb_mean:  0.0, ssd_write_bw_mb_mean:  0.0, ssd_read_iops_mean:    0, ssd_util_pct_mean:  0.1, kv_cache_hit_rate_pct:  2 },
  { inference_concurrency: 16, ttft_mean_ms:  8900, ttft_p99_ms: 24000, tpot_mean_ms:  415, tpot_p99_ms: 1800, throughput_req_s: 0.17, throughput_tok_s:  47, prompt_tokens_mean: 4810, output_tokens_mean:  285, hbm_used_gb_mean: 210, hbm_util_pct_mean:  66, dram_used_gb_mean: 52, dram_util_pct_mean: 10, ssd_read_bw_mb_mean:  1.2, ssd_write_bw_mb_mean:  0.8, ssd_read_iops_mean:  280, ssd_util_pct_mean:  3.4, kv_cache_hit_rate_pct:  8 },
  { inference_concurrency: 32, ttft_mean_ms: 16200, ttft_p99_ms: 48000, tpot_mean_ms:  580, tpot_p99_ms: 3400, throughput_req_s: 0.28, throughput_tok_s:  72, prompt_tokens_mean: 4795, output_tokens_mean:  261, hbm_used_gb_mean: 254, hbm_util_pct_mean:  79, dram_used_gb_mean: 58, dram_util_pct_mean: 11, ssd_read_bw_mb_mean: 42.8, ssd_write_bw_mb_mean: 38.4, ssd_read_iops_mean: 9800, ssd_util_pct_mean: 48.2, kv_cache_hit_rate_pct: 24 },
  { inference_concurrency: 52, ttft_mean_ms: 31000, ttft_p99_ms: 92000, tpot_mean_ms:  920, tpot_p99_ms: 6800, throughput_req_s: 0.31, throughput_tok_s:  78, prompt_tokens_mean: 4802, output_tokens_mean:  249, hbm_used_gb_mean: 256, hbm_util_pct_mean:  80, dram_used_gb_mean: 72, dram_util_pct_mean: 14, ssd_read_bw_mb_mean: 98.6, ssd_write_bw_mb_mean: 89.2, ssd_read_iops_mean:22400, ssd_util_pct_mean: 87.4, kv_cache_hit_rate_pct: 38 },
];

// ── CSV parser ─────────────────────────────────────────────────────────────────
function parseCSV(text) {
  const lines = text.trim().split("\n").filter(Boolean);
  if (lines.length < 2) return null;
  const headers = lines[0].split(",").map(h => h.trim());
  return lines.slice(1).map(line => {
    const vals = line.split(",");
    const obj = {};
    headers.forEach((h, i) => {
      const v = vals[i]?.trim();
      obj[h] = isNaN(v) ? v : Number(v);
    });
    return obj;
  });
}

// ── Color palette ─────────────────────────────────────────────────────────────
const C = {
  hbm:    "#22d3ee",
  dram:   "#a78bfa",
  ssd:    "#fb923c",
  ttft:   "#f472b6",
  tpot:   "#4ade80",
  req:    "#facc15",
  iops:   "#fb923c",
  bg:     "#0a0e1a",
  panel:  "#0f1628",
  border: "#1e2d4a",
  text:   "#e2e8f0",
  muted:  "#64748b",
  evict:  "#ef4444",
};

// ── Tooltip ────────────────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label, formatter }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0f1628", border: "1px solid #1e2d4a",
      borderRadius: 8, padding: "10px 14px", fontSize: 12,
      boxShadow: "0 4px 24px #000a"
    }}>
      <div style={{ color: "#94a3b8", marginBottom: 6, fontFamily: "monospace" }}>
        concurrency = {label}
      </div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{formatter ? formatter(p.value, p.name) : p.value}</strong>
        </div>
      ))}
    </div>
  );
};

// ── Metric card ────────────────────────────────────────────────────────────────
const MetricCard = ({ label, value, unit, color, sub }) => (
  <div style={{
    background: C.panel, border: `1px solid ${C.border}`,
    borderLeft: `3px solid ${color}`, borderRadius: 8,
    padding: "12px 16px", minWidth: 140
  }}>
    <div style={{ color: C.muted, fontSize: 11, textTransform: "uppercase",
                  letterSpacing: 1, marginBottom: 4 }}>{label}</div>
    <div style={{ color, fontSize: 22, fontWeight: 700, fontFamily: "monospace" }}>
      {value}<span style={{ fontSize: 13, color: C.muted, marginLeft: 4 }}>{unit}</span>
    </div>
    {sub && <div style={{ color: C.muted, fontSize: 11, marginTop: 2 }}>{sub}</div>}
  </div>
);

// ── Section header ─────────────────────────────────────────────────────────────
const SectionHeader = ({ title, note }) => (
  <div style={{ marginBottom: 12, marginTop: 28 }}>
    <div style={{ fontSize: 13, fontWeight: 600, color: C.text,
                  textTransform: "uppercase", letterSpacing: 1.5 }}>{title}</div>
    {note && <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>{note}</div>}
  </div>
);

const fmt = (v, unit) => `${typeof v === "number" ? v.toLocaleString() : v} ${unit}`;
const fmtMs = v => `${(v/1000).toFixed(1)}s`;

// ── Main component ─────────────────────────────────────────────────────────────
export default function SweepDashboard() {
  const [data, setData] = useState(DEMO_DATA);
  const [rawCSV, setRawCSV] = useState("");
  const [showImport, setShowImport] = useState(false);
  const [parseError, setParseError] = useState("");

  const handleCSV = useCallback(() => {
    const parsed = parseCSV(rawCSV);
    if (!parsed || parsed.length === 0) {
      setParseError("Could not parse CSV — check format");
      return;
    }
    setData(parsed);
    setParseError("");
    setShowImport(false);
  }, [rawCSV]);

  // Eviction threshold line
  const evictThreshold = 26;
  const lastPoint = data[data.length - 1] || {};

  return (
    <div style={{
      background: C.bg, color: C.text, minHeight: "100vh",
      fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
      padding: "24px 28px", boxSizing: "border-box"
    }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start",
                    justifyContent: "space-between", marginBottom: 28 }}>
        <div>
          <div style={{ fontSize: 20, fontWeight: 700, color: "#e2e8f0",
                        letterSpacing: 1 }}>
            AMOprof · Concurrency Sweep Dashboard
          </div>
          <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>
            SWE-bench inference_concurrency → TTFT · TPOT · Throughput · HBM · DRAM · SSD
          </div>
        </div>
        <button onClick={() => setShowImport(v => !v)} style={{
          background: showImport ? "#1e3a5f" : "transparent",
          border: "1px solid #1e2d4a", color: "#94a3b8",
          borderRadius: 6, padding: "6px 14px", cursor: "pointer",
          fontSize: 12, fontFamily: "inherit"
        }}>
          {showImport ? "Close" : "Import CSV"}
        </button>
      </div>

      {/* CSV import */}
      {showImport && (
        <div style={{ background: C.panel, border: `1px solid ${C.border}`,
                      borderRadius: 10, padding: 16, marginBottom: 24 }}>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 8 }}>
            Paste contents of <code>concurrency_sweep.csv</code> from AMOprof output:
          </div>
          <textarea value={rawCSV} onChange={e => setRawCSV(e.target.value)}
            placeholder="inference_concurrency,ttft_mean_ms,ttft_p99_ms,..."
            style={{
              width: "100%", height: 120, background: "#0a0e1a",
              border: "1px solid #1e2d4a", borderRadius: 6, color: "#e2e8f0",
              padding: 10, fontFamily: "monospace", fontSize: 11, boxSizing: "border-box",
              resize: "vertical"
            }} />
          {parseError && <div style={{ color: C.evict, fontSize: 11, marginTop: 4 }}>{parseError}</div>}
          <button onClick={handleCSV} style={{
            marginTop: 10, background: "#1e3a5f", border: "1px solid #2563eb",
            color: "#93c5fd", borderRadius: 6, padding: "6px 16px",
            cursor: "pointer", fontFamily: "inherit", fontSize: 12
          }}>Load CSV →</button>
        </div>
      )}

      {/* Eviction note */}
      <div style={{
        background: "#1a0a00", border: "1px solid #7c2d12",
        borderRadius: 8, padding: "10px 16px", marginBottom: 24,
        fontSize: 12, color: "#fdba74"
      }}>
        ⚡ HiCache eviction to SSD starts at concurrency ≥ 26
        (8× A100 40GB, R1-70B, 65K ctx: 26 × ~10 GB = 260 GB &gt; 256 GB HBM KV pool)
      </div>

      {/* Summary cards — last measured point */}
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 28 }}>
        <MetricCard label="Peak TTFT" value={lastPoint.ttft_mean_ms?.toLocaleString()} unit="ms"
          color={C.ttft} sub={`P99: ${lastPoint.ttft_p99_ms?.toLocaleString()} ms`} />
        <MetricCard label="Peak TPOT" value={lastPoint.tpot_mean_ms} unit="ms"
          color={C.tpot} sub={`P99: ${lastPoint.tpot_p99_ms} ms`} />
        <MetricCard label="Throughput" value={lastPoint.throughput_req_s?.toFixed(2)} unit="req/s"
          color={C.req} sub={`${lastPoint.throughput_tok_s?.toFixed(0)} tok/s`} />
        <MetricCard label="HBM util" value={lastPoint.hbm_util_pct_mean?.toFixed(0)} unit="%"
          color={C.hbm} sub={`${lastPoint.hbm_used_gb_mean?.toFixed(0)} GB used`} />
        <MetricCard label="SSD read BW" value={lastPoint.ssd_read_bw_mb_mean?.toFixed(1)} unit="MB/s"
          color={C.ssd} sub={`write: ${lastPoint.ssd_write_bw_mb_mean?.toFixed(1)} MB/s`} />
        <MetricCard label="SSD IOPS" value={lastPoint.ssd_read_iops_mean?.toLocaleString()} unit=""
          color={C.iops} sub={`util: ${lastPoint.ssd_util_pct_mean?.toFixed(1)}%`} />
      </div>

      {/* ── Row 1: Latency ─────────────────────────────────────────────────── */}
      <SectionHeader title="Inference Latency vs Concurrency"
        note="TTFT and TPOT increase as server queues fill — SSD eviction adds KV$ restore stall" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 8 }}>

        <div style={{ background: C.panel, border: `1px solid ${C.border}`,
                      borderRadius: 10, padding: 16 }}>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 12 }}>
            TTFT — Time to First Token (ms)
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <defs>
                <linearGradient id="gTTFT" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={C.ttft} stopOpacity={0.25}/>
                  <stop offset="95%" stopColor={C.ttft} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
              <XAxis dataKey="inference_concurrency" stroke={C.muted} tick={{ fontSize: 11 }}
                label={{ value: "concurrency", position: "insideBottom", offset: -2, fill: C.muted, fontSize: 11 }} />
              <YAxis stroke={C.muted} tick={{ fontSize: 11 }}
                tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}s` : v} />
              <Tooltip content={<CustomTooltip formatter={(v) => fmtMs(v)} />} />
              <ReferenceLine x={evictThreshold} stroke={C.evict} strokeDasharray="4 4"
                label={{ value: "SSD eviction", fill: C.evict, fontSize: 10, position: "top" }} />
              <Area type="monotone" dataKey="ttft_mean_ms" stroke={C.ttft}
                fill="url(#gTTFT)" strokeWidth={2} name="TTFT mean" dot={{ r: 3 }} />
              <Area type="monotone" dataKey="ttft_p99_ms" stroke="#f9a8d4"
                fill="none" strokeWidth={1.5} strokeDasharray="4 2" name="TTFT P99" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div style={{ background: C.panel, border: `1px solid ${C.border}`,
                      borderRadius: 10, padding: 16 }}>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 12 }}>
            TPOT — Time Per Output Token / ITL (ms)
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <defs>
                <linearGradient id="gTPOT" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={C.tpot} stopOpacity={0.25}/>
                  <stop offset="95%" stopColor={C.tpot} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
              <XAxis dataKey="inference_concurrency" stroke={C.muted} tick={{ fontSize: 11 }}
                label={{ value: "concurrency", position: "insideBottom", offset: -2, fill: C.muted, fontSize: 11 }} />
              <YAxis stroke={C.muted} tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip formatter={(v) => `${v.toFixed(1)} ms`} />} />
              <ReferenceLine x={evictThreshold} stroke={C.evict} strokeDasharray="4 4" />
              <Area type="monotone" dataKey="tpot_mean_ms" stroke={C.tpot}
                fill="url(#gTPOT)" strokeWidth={2} name="TPOT mean" dot={{ r: 3 }} />
              <Area type="monotone" dataKey="tpot_p99_ms" stroke="#86efac"
                fill="none" strokeWidth={1.5} strokeDasharray="4 2" name="TPOT P99" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Row 2: Throughput + Tokens ────────────────────────────────────── */}
      <SectionHeader title="Throughput and Token Counts"
        note="Throughput peaks then plateaus as queuing and SSD stall dominate over parallelism gains" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 8 }}>

        <div style={{ background: C.panel, border: `1px solid ${C.border}`,
                      borderRadius: 10, padding: 16 }}>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 12 }}>
            Throughput — req/s and tok/s
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
              <XAxis dataKey="inference_concurrency" stroke={C.muted} tick={{ fontSize: 11 }} />
              <YAxis yAxisId="req" stroke={C.req} tick={{ fontSize: 11 }} />
              <YAxis yAxisId="tok" orientation="right" stroke={C.tpot} tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLine yAxisId="req" x={evictThreshold} stroke={C.evict} strokeDasharray="4 4" />
              <Line yAxisId="req" type="monotone" dataKey="throughput_req_s"
                stroke={C.req} strokeWidth={2} name="req/s" dot={{ r: 3 }} />
              <Line yAxisId="tok" type="monotone" dataKey="throughput_tok_s"
                stroke={C.tpot} strokeWidth={2} name="tok/s" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ background: C.panel, border: `1px solid ${C.border}`,
                      borderRadius: 10, padding: 16 }}>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 12 }}>
            Token counts per instance (mean)
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
              <XAxis dataKey="inference_concurrency" stroke={C.muted} tick={{ fontSize: 11 }} />
              <YAxis stroke={C.muted} tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="prompt_tokens_mean" fill="#38bdf8" name="prompt tok" radius={[3,3,0,0]} />
              <Bar dataKey="output_tokens_mean" fill="#818cf8" name="output tok" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Row 3: Memory tiers ────────────────────────────────────────────── */}
      <SectionHeader title="Memory Tier Utilisation"
        note="L1:HBM fills, then L2:DRAM staging grows, then L3:SSD KV$ eviction begins" />
      <div style={{ background: C.panel, border: `1px solid ${C.border}`,
                    borderRadius: 10, padding: 16, marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: C.muted, marginBottom: 12 }}>
          HBM used (GB) · DRAM used (GB) · SSD BW (MB/s)
          <span style={{ marginLeft: 16, color: C.evict }}>
            — eviction starts at concurrency ≈ 26
          </span>
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={data} margin={{ top: 4, right: 40, bottom: 4, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
            <XAxis dataKey="inference_concurrency" stroke={C.muted} tick={{ fontSize: 11 }}
              label={{ value: "inference_concurrency", position: "insideBottom", offset: -2, fill: C.muted, fontSize: 11 }} />
            <YAxis yAxisId="gb" stroke={C.hbm} tick={{ fontSize: 11 }}
              label={{ value: "GB", angle: -90, position: "insideLeft", fill: C.hbm, fontSize: 11 }} />
            <YAxis yAxisId="bw" orientation="right" stroke={C.ssd} tick={{ fontSize: 11 }}
              label={{ value: "MB/s", angle: 90, position: "insideRight", fill: C.ssd, fontSize: 11 }} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine yAxisId="gb" x={evictThreshold} stroke={C.evict}
              strokeDasharray="4 4"
              label={{ value: "SSD eviction threshold", fill: C.evict, fontSize: 10, position: "insideTopLeft" }} />
            <Line yAxisId="gb" type="monotone" dataKey="hbm_used_gb_mean"
              stroke={C.hbm} strokeWidth={2.5} name="L1:HBM used (GB)" dot={{ r: 4 }} />
            <Line yAxisId="gb" type="monotone" dataKey="dram_used_gb_mean"
              stroke={C.dram} strokeWidth={2} name="L2:DRAM used (GB)" dot={{ r: 3 }} />
            <Line yAxisId="bw" type="monotone" dataKey="ssd_read_bw_mb_mean"
              stroke={C.ssd} strokeWidth={2} name="L3:SSD read BW (MB/s)" dot={{ r: 3 }} />
            <Line yAxisId="bw" type="monotone" dataKey="ssd_write_bw_mb_mean"
              stroke="#f97316" strokeWidth={2} strokeDasharray="5 3"
              name="L3:SSD write BW (MB/s)" dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── Row 4: SSD detail ──────────────────────────────────────────────── */}
      <SectionHeader title="SSD Metrics — HiCache KV$ Eviction Detail"
        note="IOPS, utilisation and KV$ cache hit rate — only meaningful when concurrency ≥ 26" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 8 }}>

        <div style={{ background: C.panel, border: `1px solid ${C.border}`,
                      borderRadius: 10, padding: 16 }}>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 12 }}>
            SSD Read IOPS + Utilisation %
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
              <XAxis dataKey="inference_concurrency" stroke={C.muted} tick={{ fontSize: 11 }} />
              <YAxis yAxisId="iops" stroke={C.iops} tick={{ fontSize: 11 }} />
              <YAxis yAxisId="util" orientation="right" stroke={C.muted} tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLine yAxisId="iops" x={evictThreshold} stroke={C.evict} strokeDasharray="4 4" />
              <Line yAxisId="iops" type="monotone" dataKey="ssd_read_iops_mean"
                stroke={C.iops} strokeWidth={2} name="read IOPS" dot={{ r: 3 }} />
              <Line yAxisId="util" type="monotone" dataKey="ssd_util_pct_mean"
                stroke="#fcd34d" strokeWidth={2} name="util %" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ background: C.panel, border: `1px solid ${C.border}`,
                      borderRadius: 10, padding: 16 }}>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 12 }}>
            KV$ Cache Hit Rate % (RadixAttention prefix reuse)
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <defs>
                <linearGradient id="gHit" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#22d3ee" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
              <XAxis dataKey="inference_concurrency" stroke={C.muted} tick={{ fontSize: 11 }} />
              <YAxis stroke={C.muted} tick={{ fontSize: 11 }} domain={[0, 100]} />
              <Tooltip content={<CustomTooltip formatter={(v) => `${v.toFixed(1)}%`} />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Area type="monotone" dataKey="kv_cache_hit_rate_pct"
                stroke="#22d3ee" fill="url(#gHit)" strokeWidth={2}
                name="KV$ hit rate %" dot={{ r: 3 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Data table ─────────────────────────────────────────────────────── */}
      <SectionHeader title="Raw Data Table" />
      <div style={{ overflowX: "auto", marginBottom: 32 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11,
                        fontFamily: "monospace" }}>
          <thead>
            <tr style={{ background: "#0f1628" }}>
              {["concurrency","TTFT mean","TTFT P99","TPOT mean","TPOT P99",
                "req/s","tok/s","HBM GB","HBM %","DRAM GB","SSD rBW","SSD wBW","IOPS","SSD util %","KV$ hit%"].map(h => (
                <th key={h} style={{ padding: "8px 12px", textAlign: "right",
                                     color: C.muted, borderBottom: `1px solid ${C.border}`,
                                     whiteSpace: "nowrap" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => {
              const isEvict = row.inference_concurrency >= evictThreshold;
              return (
                <tr key={i} style={{
                  background: isEvict ? "#1a0a00" : (i % 2 === 0 ? C.panel : C.bg),
                  borderLeft: isEvict ? `3px solid ${C.evict}` : `3px solid transparent`
                }}>
                  {[
                    [row.inference_concurrency, C.text],
                    [row.ttft_mean_ms?.toLocaleString()+" ms", C.ttft],
                    [row.ttft_p99_ms?.toLocaleString()+" ms", "#f9a8d4"],
                    [row.tpot_mean_ms?.toFixed(1)+" ms", C.tpot],
                    [row.tpot_p99_ms?.toFixed(0)+" ms", "#86efac"],
                    [row.throughput_req_s?.toFixed(3), C.req],
                    [row.throughput_tok_s?.toFixed(0), C.req],
                    [row.hbm_used_gb_mean?.toFixed(0)+" GB", C.hbm],
                    [row.hbm_util_pct_mean?.toFixed(0)+"%", C.hbm],
                    [row.dram_used_gb_mean?.toFixed(0)+" GB", C.dram],
                    [row.ssd_read_bw_mb_mean?.toFixed(1)+" MB/s", C.ssd],
                    [row.ssd_write_bw_mb_mean?.toFixed(1)+" MB/s", "#f97316"],
                    [row.ssd_read_iops_mean?.toLocaleString(), C.iops],
                    [row.ssd_util_pct_mean?.toFixed(1)+"%", C.iops],
                    [row.kv_cache_hit_rate_pct?.toFixed(1)+"%", "#22d3ee"],
                  ].map(([val, color], j) => (
                    <td key={j} style={{ padding: "7px 12px", textAlign: "right",
                                         color, borderBottom: `1px solid ${C.border}` }}>
                      {val}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: 14,
                    color: C.muted, fontSize: 11 }}>
        AMOprof — AI Workload-Aware Storage Profiler · Concurrency sweep data ·
        Red rows = HiCache SSD eviction active (concurrency ≥ {evictThreshold}) ·
        Demo data shown — import your concurrency_sweep.csv for real results
      </div>
    </div>
  );
}
