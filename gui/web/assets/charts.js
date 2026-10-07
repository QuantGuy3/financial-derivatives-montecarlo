// Constructores de opciones de ECharts. Los colores salen de las variables CSS (claro/oscuro) y las
// opciones se reconstruyen al cambiar de tema, de idioma o de datos.

export function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n) => cs.getPropertyValue(n).trim();
  return {
    text: g('--text'), text2: g('--text-2'), muted: g('--muted'), grid: g('--grid'), axis: g('--axis'),
    surface: g('--surface'), accent: g('--accent'), good: g('--good'), crit: g('--crit'), warn: g('--warn'),
    series: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => g(`--series-${i}`)),
  };
}

export const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export function fmtNum(x, digits = 4) {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—';
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e7)) return x.toExponential(2);
  return x.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: Math.min(2, digits) }).replace(/,/g, ' ');
}

// Decimales razonables para un valor con incertidumbre `se`
export function decimalsFor(se) {
  if (!Number.isFinite(se) || se <= 0) return 4;
  return Math.max(2, Math.min(8, Math.ceil(-Math.log10(se)) + 1));
}

export function fmtInt(n) {
  if (!Number.isFinite(n)) return '—';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' G';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + ' M';
  if (n >= 1e4) return (n / 1e3).toFixed(1) + ' k';
  return String(Math.round(n));
}

function axisBase(c) {
  return {
    axisLine: { lineStyle: { color: c.axis } },
    axisTick: { lineStyle: { color: c.axis } },
    axisLabel: { color: c.muted, fontSize: 11 },
    splitLine: { lineStyle: { color: c.grid } },
    nameTextStyle: { color: c.text2, fontSize: 11.5 },
  };
}

function baseOption(c, anim) {
  return {
    animation: anim,
    animationDuration: 500,
    animationDurationUpdate: anim ? 280 : 0,
    animationEasing: 'cubicOut',
    animationEasingUpdate: 'cubicOut',
    textStyle: { fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: c.text2 },
    backgroundColor: 'transparent',
    legend: { top: 2, right: 8, textStyle: { color: c.text2, fontSize: 11.5 }, itemWidth: 14, itemHeight: 3, icon: 'roundRect', inactiveColor: c.axis },
    tooltip: {
      trigger: 'axis', confine: true, backgroundColor: c.surface, borderColor: c.axis, borderWidth: 1,
      textStyle: { color: c.text, fontSize: 12 }, extraCssText: 'box-shadow:0 6px 24px rgba(0,0,0,.18);border-radius:10px;',
      axisPointer: { type: 'line', lineStyle: { color: c.muted, type: 'dashed', width: 1 } },
    },
    toolbox: {
      right: 8, top: 26, itemSize: 14, iconStyle: { borderColor: c.muted },
      emphasis: { iconStyle: { borderColor: c.accent } },
      feature: { saveAsImage: { pixelRatio: 2, backgroundColor: c.surface }, dataZoom: { yAxisIndex: 'none' }, restore: {} },
    },
  };
}

// ---- Convergencia: estimación del precio con banda ±2σ -------------------------------------------------

export function priceOption(d, c, anim, t) {
  const pts = d.pts;                                   // [n, mean, se]
  const o = baseOption(c, anim);
  const last = pts[pts.length - 1];
  let lo = 0, hi = 1;
  if (last) {
    const seL = Math.max(last[2], 1e-12);
    const center = d.ref !== null && d.ref !== undefined ? d.ref : last[1];
    const half = Math.max(d.eps ? 6 * d.eps : 0, 6 * seL, 1e-9);
    lo = center - half; hi = center + half;
    if (d.ref !== null && d.ref !== undefined) { lo = Math.min(lo, last[1] - 3 * seL); hi = Math.max(hi, last[1] + 3 * seL); }
  }
  const dec = decimalsFor(last ? last[2] : 0.01);
  const lower = pts.map((p) => [p[0], p[1] - 2 * p[2]]);
  const width = pts.map((p) => [p[0], 4 * p[2]]);
  const est = pts.map((p) => [p[0], p[1]]);

  const refSeries = [];
  if (d.ref !== null && d.ref !== undefined) {
    refSeries.push({
      name: t('legend.ref'), type: 'line', data: [], silent: true, lineStyle: { color: c.text2, type: 'dashed', width: 1.4 }, itemStyle: { color: c.text2 },
      markLine: {
        symbol: 'none', silent: true, lineStyle: { color: c.text2, type: 'dashed', width: 1.4 },
        label: { color: c.text2, fontSize: 11, formatter: () => `${t('legend.ref')} ${d.ref.toFixed(dec)}`, position: 'insideEndTop' },
        data: [{ yAxis: d.ref }],
      },
      markArea: d.eps ? {
        silent: true, itemStyle: { color: c.good, opacity: 0.09 },
        data: [[{ yAxis: d.ref - d.eps }, { yAxis: d.ref + d.eps }]],
      } : undefined,
    });
  }

  return {
    ...o,
    grid: { left: 62, right: 22, top: 56, bottom: 62 },
    legend: { ...o.legend, data: [t('legend.est'), t('legend.band'), ...(d.ref !== null && d.ref !== undefined ? [t('legend.ref')] : [])] },
    tooltip: {
      ...o.tooltip,
      formatter: (ps) => {
        const p = ps.find((x) => x.seriesName === t('legend.est'));
        if (!p) return '';
        const [n, m] = p.data;
        const idx = p.dataIndex;
        const se = pts[idx] ? pts[idx][2] : 0;
        let h = `<div style="font-weight:650;margin-bottom:4px">N = ${Math.round(n).toLocaleString('en-US').replace(/,/g, ' ')}</div>`;
        h += `<div><span style="display:inline-block;width:12px;height:3px;background:${c.series[0]};border-radius:2px;margin-right:6px;vertical-align:middle"></span><b>${m.toFixed(dec)}</b> <span style="color:${c.muted}">±${(2 * se).toFixed(dec)} (2σ)</span></div>`;
        if (d.ref !== null && d.ref !== undefined) h += `<div style="color:${c.muted}">${t('tip.err')} ${(m - d.ref >= 0 ? '+' : '')}${(m - d.ref).toFixed(dec)}</div>`;
        return h;
      },
    },
    xAxis: { type: 'log', name: t('axis.n'), nameLocation: 'middle', nameGap: 30, min: 'dataMin', ...axisBase(c),
      axisLabel: { ...axisBase(c).axisLabel, formatter: (v) => fmtInt(v) } },
    yAxis: { type: 'value', min: lo, max: hi, scale: true, ...axisBase(c),
      axisLabel: { ...axisBase(c).axisLabel, formatter: (v) => v.toFixed(Math.max(2, dec - 1)) } },
    dataZoom: [{ type: 'inside', xAxisIndex: 0, filterMode: 'none' }, { type: 'slider', xAxisIndex: 0, height: 14, bottom: 8, filterMode: 'none',
      borderColor: 'transparent', backgroundColor: c.grid, fillerColor: 'rgba(42,120,214,.18)', handleSize: '90%', textStyle: { color: c.muted } }],
    series: [
      { name: '_lo', type: 'line', stack: 'ci', data: lower, symbol: 'none', lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
      { name: t('legend.band'), type: 'line', stack: 'ci', data: width, symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: c.series[0], opacity: 0.16 },
        itemStyle: { color: c.series[0], opacity: 0.45 }, silent: true },
      { name: t('legend.est'), type: 'line', data: est, symbol: 'circle', symbolSize: 5, showSymbol: pts.length < 40,
        lineStyle: { width: 2, color: c.series[0] }, itemStyle: { color: c.series[0], borderColor: c.surface, borderWidth: 1.5 }, z: 5,
        emphasis: { focus: 'none' } },
      ...refSeries,
    ],
  };
}

// ---- Convergencia: error estándar y error real frente a N (log-log) ----------------------------------------

export function errorOption(d, c, anim, t) {
  const pts = d.pts.filter((p) => p[2] > 0 && p[0] > 0);
  const o = baseOption(c, anim);
  const se = pts.map((p) => [p[0], p[2]]);
  const err = d.ref !== null && d.ref !== undefined
    ? d.pts.filter((p) => p[0] > 0).map((p) => [p[0], Math.max(Math.abs(p[1] - d.ref), 1e-12)]) : [];
  const guides = [];
  if (pts.length >= 1) {
    const first = pts[Math.min(pts.length - 1, Math.max(0, Math.floor(pts.length * 0.12)))];
    const n0 = first[0], s0 = first[1] > 0 ? first[2] : 1;
    const nMax = pts[pts.length - 1][0] * 8;
    const mk = (name, slope, color) => ({
      name, type: 'line', silent: true, symbol: 'none', lineStyle: { type: 'dashed', width: 1.2, color },
      data: [[n0, s0], [nMax, s0 * Math.pow(nMax / n0, -slope)]], z: 1,
      endLabel: { show: true, formatter: name, color, fontSize: 10.5, distance: 4 }, tooltip: { show: false },
    });
    guides.push(mk('∝ N⁻¹ᐟ²', 0.5, c.muted));
    guides.push(mk('∝ N⁻¹', 1, c.series[2]));
  }
  const target = d.eps ? [{
    name: t('legend.target'), type: 'line', data: [], silent: true,
    markLine: { symbol: 'none', silent: true, lineStyle: { color: c.warn, type: 'dotted', width: 1.6 },
      label: { color: c.text2, fontSize: 11, formatter: 'ε/√2', position: 'insideStartTop' }, data: [{ yAxis: d.eps / Math.SQRT2 }] },
  }] : [];

  return {
    ...o,
    grid: { left: 66, right: 54, top: 56, bottom: 46 },
    legend: { ...o.legend, data: [t('legend.se'), ...(err.length ? [t('legend.abserr')] : [])] },
    tooltip: {
      ...o.tooltip,
      formatter: (ps) => {
        const p = ps.find((x) => x.seriesName === t('legend.se')) || ps[0];
        if (!p || !p.data) return '';
        let h = `<div style="font-weight:650;margin-bottom:4px">N = ${Math.round(p.data[0]).toLocaleString('en-US').replace(/,/g, ' ')}</div>`;
        h += `<div>${t('legend.se')}: <b>${p.data[1].toExponential(2)}</b></div>`;
        const e = ps.find((x) => x.seriesName === t('legend.abserr'));
        if (e) h += `<div style="color:${c.muted}">${t('legend.abserr')}: ${e.data[1].toExponential(2)}</div>`;
        return h;
      },
    },
    xAxis: { type: 'log', name: t('axis.n'), nameLocation: 'middle', nameGap: 28, min: 'dataMin', ...axisBase(c),
      axisLabel: { ...axisBase(c).axisLabel, formatter: (v) => fmtInt(v) } },
    yAxis: { type: 'log', ...axisBase(c), name: '', axisLabel: { ...axisBase(c).axisLabel, formatter: (v) => v.toExponential(0) } },
    series: [
      { name: t('legend.se'), type: 'line', data: se, symbol: 'circle', symbolSize: 5, showSymbol: se.length < 40,
        lineStyle: { width: 2, color: c.series[0] }, itemStyle: { color: c.series[0], borderColor: c.surface, borderWidth: 1.5 }, z: 4 },
      ...(err.length ? [{ name: t('legend.abserr'), type: 'line', data: err, symbol: 'circle', symbolSize: 4, showSymbol: err.length < 60,
        lineStyle: { width: 1.2, color: c.series[1], opacity: 0.7 }, itemStyle: { color: c.series[1] }, z: 3 }] : []),
      ...guides, ...target,
    ],
  };
}

// ---- Trayectorias ---------------------------------------------------------------------------------------------------

export function pathColors(c, n) {
  // Hasta 8 trayectorias: un color de la paleta por camino, en orden fijo. Más: un único tono tenue.
  if (n <= 8) return Array.from({ length: n }, (_, i) => ({ color: c.series[i], width: 1.9, opacity: 1 }));
  return Array.from({ length: n }, () => ({ color: c.series[0], width: 1.1, opacity: Math.max(0.22, Math.min(0.6, 14 / n)) }));
}

function niceRange(lo, hi) {
  const pad = (hi - lo) * 0.06 || 1;
  return [lo - pad, hi + pad];
}

export function pathsYRange(data, fan, marks) {
  let lo = Infinity, hi = -Infinity;
  const take = (arr) => { for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; } };
  for (const p of data.paths) take(p.S);
  if (data.coupled) { take(data.coupled.S_fine); }
  if (fan) { take(fan.p05); take(fan.p95); }
  for (const k of ['strike', 'barrier', 'S0']) if (marks && marks[k] !== undefined && Number.isFinite(marks[k])) { lo = Math.min(lo, marks[k]); hi = Math.max(hi, marks[k]); }
  return niceRange(lo, hi);
}

// view: { data, k, fan, showFan, showMean, showRun, coupled, mu, T, yr }
export function pathsOption(view, c, anim, t) {
  const o = baseOption(c, anim);
  const { data, k } = view;
  const tt = data.t;
  const n = tt.length - 1;
  const kk = Math.max(0, Math.min(n, k));
  const series = [];

  if (view.coupled && data.coupled) {
    const cp = data.coupled;
    const kf = Math.min(cp.t_fine.length - 1, Math.round(kk * (cp.t_fine.length - 1) / n));
    const kc = Math.min(cp.t_coarse.length - 1, Math.floor(kf / (cp.t_fine.length - 1) * (cp.t_coarse.length - 1)));
    series.push({ name: `${t('legend.fine')} (${cp.t_fine.length - 1})`, type: 'line', data: cp.t_fine.slice(0, kf + 1).map((x, i) => [x, cp.S_fine[i]]),
      symbol: 'none', lineStyle: { width: 1.8, color: c.series[0] }, itemStyle: { color: c.series[0] }, z: 4 });
    series.push({ name: `${t('legend.coarse')} (${cp.t_coarse.length - 1})`, type: 'line', step: 'end', data: cp.t_coarse.slice(0, kc + 1).map((x, i) => [x, cp.S_coarse[i]]),
      symbol: 'circle', symbolSize: 6, lineStyle: { width: 2.2, color: c.series[1] }, itemStyle: { color: c.series[1] }, z: 5 });
  } else {
    if (view.showFan && view.fan) {
      const f = view.fan;
      const band = (lo, hi, name, opacity) => ({
        name, type: 'custom', silent: true, z: 0, data: [0], tooltip: { show: false }, itemStyle: { color: c.series[0], opacity },
        renderItem: (params, api) => {
          const pts = [];
          for (let i = 0; i < f.t.length; i++) pts.push(api.coord([f.t[i], f[hi][i]]));
          for (let i = f.t.length - 1; i >= 0; i--) pts.push(api.coord([f.t[i], f[lo][i]]));
          return { type: 'polygon', shape: { points: pts }, style: { fill: c.series[0], opacity } };
        },
      });
      series.push(band('p05', 'p95', t('legend.fan90'), 0.10), band('p25', 'p75', t('legend.fan50'), 0.16));
    }
    if (view.showMean) {
      const h = view.T / n;
      const mean = tt.map((x, i) => [x, view.S0 * Math.pow(1 + view.mu * h, i)]);
      series.push({ name: t('legend.mean'), type: 'line', data: mean, symbol: 'none', silent: true,
        lineStyle: { width: 1.5, type: 'dashed', color: c.text2 }, itemStyle: { color: c.text2 }, z: 2 });
    }
    const styles = pathColors(c, data.paths.length);
    data.paths.forEach((p, i) => {
      const s = styles[i];
      const hit = (data.marks && data.marks.barrier !== undefined && p.run) ? p.run.findIndex((v) => v >= data.marks.barrier) : -1;
      series.push({
        name: `${t('legend.path')} ${i + 1}`, type: 'line', data: p.S.slice(0, kk + 1).map((v, j) => [tt[j], v]),
        symbol: 'none', lineStyle: { width: s.width, color: s.color, opacity: s.opacity }, itemStyle: { color: s.color }, z: 3,
        emphasis: { focus: 'series', lineStyle: { width: s.width + 1.2, opacity: 1 } },
        markPoint: hit >= 1 && hit <= kk ? { symbol: 'pin', symbolSize: 26, itemStyle: { color: c.crit },
          label: { show: true, formatter: '✕', color: '#fff', fontSize: 11 }, data: [{ coord: [tt[hit], p.S[hit]] }] } : undefined,
      });
      if (view.showRun && i === 0 && p.run) {
        series.push({ name: t('legend.run'), type: 'line', data: p.run.slice(0, kk + 1).map((v, j) => [tt[j], v]),
          symbol: 'none', lineStyle: { width: 1.5, type: 'dotted', color: c.series[3] }, itemStyle: { color: c.series[3] }, z: 2 });
      }
    });
    if (kk === n && data.paths.length <= 8) {
      series.push({ name: '_end', type: 'scatter', silent: true, symbolSize: 9, z: 6,
        data: data.paths.map((p, i) => ({ value: [tt[n], p.S[n]], itemStyle: { color: styles[i].color, borderColor: c.surface, borderWidth: 2 },
          label: { show: true, position: 'right', formatter: p.payoff > 0 ? `Y=${p.payoff.toFixed(2)}` : 'Y=0', color: c.text2, fontSize: 10.5 } })) });
    }
  }

  // Líneas de referencia del payoff
  const lines = [];
  const m = data.marks || {};
  if (m.strike !== undefined) lines.push({ yAxis: m.strike, lineStyle: { color: c.series[2], type: 'dashed', width: 1.3 },
    label: { formatter: `K = ${m.strike}`, color: c.text2, fontSize: 11, position: 'insideStartTop' } });
  if (m.barrier !== undefined) lines.push({ yAxis: m.barrier, lineStyle: { color: c.crit, type: 'solid', width: 1.8 },
    label: { formatter: `B = ${m.barrier}`, color: c.crit, fontSize: 11, position: 'insideStartTop' } });
  if (m.S0 !== undefined) lines.push({ yAxis: m.S0, lineStyle: { color: c.muted, type: 'dotted', width: 1 },
    label: { formatter: 'S₀', color: c.muted, fontSize: 10.5, position: 'insideEndTop' } });
  series.push({ name: '_marks', type: 'line', data: [], silent: true, markLine: { symbol: 'none', silent: true, data: lines } });

  return {
    ...o,
    grid: { left: 60, right: 78, top: 46, bottom: 34 },
    legend: { ...o.legend, show: data.paths.length <= 8 || view.coupled, type: 'scroll', data: series.filter((s) => !s.name.startsWith('_')).map((s) => s.name) },
    tooltip: { ...o.tooltip, trigger: 'axis', valueFormatter: (v) => (typeof v === 'number' ? v.toFixed(2) : v) },
    xAxis: { type: 'value', min: 0, max: view.T, name: t('axis.t'), nameLocation: 'middle', nameGap: 22, ...axisBase(c), splitLine: { show: false },
      axisLabel: { ...axisBase(c).axisLabel, formatter: (v) => v.toFixed(2) } },
    yAxis: { type: 'value', min: view.yr[0], max: view.yr[1], ...axisBase(c), axisLabel: { ...axisBase(c).axisLabel, formatter: (v) => v.toFixed(0) } },
    dataZoom: [{ type: 'inside', filterMode: 'none' }],
    series,
  };
}

export function varOption(view, c, anim, t) {
  const o = baseOption(c, anim);
  const { data, k } = view;
  const tt = data.t;
  const kk = Math.max(0, Math.min(tt.length - 1, k));
  const styles = pathColors(c, data.paths.length);
  let hi = 0;
  for (const p of data.paths) if (p.V) for (const v of p.V) if (v > hi) hi = v;
  return {
    ...o,
    legend: { show: false },
    grid: { left: 60, right: 78, top: 14, bottom: 30 },
    xAxis: { type: 'value', min: 0, max: view.T, ...axisBase(c), splitLine: { show: false }, axisLabel: { ...axisBase(c).axisLabel, formatter: (v) => v.toFixed(2) } },
    yAxis: { type: 'value', min: 0, max: hi * 1.08 || 0.1, name: 'v(t)', ...axisBase(c), axisLabel: { ...axisBase(c).axisLabel, formatter: (v) => v.toFixed(2) } },
    series: data.paths.map((p, i) => ({
      name: `${t('legend.path')} ${i + 1}`, type: 'line', symbol: 'none', z: 3,
      lineStyle: { width: styles[i].width * 0.8, color: styles[i].color, opacity: styles[i].opacity },
      data: (p.V || []).slice(0, kk + 1).map((v, j) => [tt[j], v]),
    })),
  };
}

export function histOption(view, c, anim, t) {
  const o = baseOption(c, anim);
  const h = view.fan && view.fan.hist;
  if (!h) return { ...o, series: [] };
  const bins = h.counts.length;
  const w = (h.hi - h.lo) / bins;
  const centers = h.counts.map((_, i) => h.lo + (i + 0.5) * w);
  const total = h.counts.reduce((a, b) => a + b, 0) || 1;
  const m = view.data.marks || {};
  const nearest = (v) => { let bi = 0, bd = Infinity; centers.forEach((x, i) => { const d = Math.abs(x - v); if (d < bd) { bd = d; bi = i; } }); return bi; };
  const lines = [];
  if (m.strike !== undefined && m.strike >= h.lo && m.strike <= h.hi) lines.push({ yAxis: nearest(m.strike), lineStyle: { color: c.series[2], type: 'dashed', width: 1.3 }, label: { formatter: 'K', color: c.text2, position: 'insideEndTop' } });
  if (m.barrier !== undefined && m.barrier >= h.lo && m.barrier <= h.hi) lines.push({ yAxis: nearest(m.barrier), lineStyle: { color: c.crit, width: 1.6 }, label: { formatter: 'B', color: c.crit, position: 'insideEndTop' } });
  return {
    ...o,
    legend: { show: false },
    grid: { left: 52, right: 14, top: 46, bottom: 34 },
    tooltip: { ...o.tooltip, trigger: 'item', formatter: (p) => `S<sub>T</sub> ≈ <b>${centers[p.dataIndex].toFixed(1)}</b><br/>${(100 * h.counts[p.dataIndex] / total).toFixed(1)} %` },
    xAxis: { type: 'value', ...axisBase(c), axisLabel: { show: false }, splitLine: { show: false }, name: '', },
    yAxis: { type: 'category', data: centers.map((x) => x.toFixed(0)), ...axisBase(c), splitLine: { show: false },
      axisLabel: { ...axisBase(c).axisLabel, interval: Math.ceil(bins / 8) - 1 } },
    series: [{
      type: 'bar', data: h.counts, barCategoryGap: '8%', itemStyle: { color: c.series[0], opacity: 0.55, borderRadius: [0, 3, 3, 0] },
      emphasis: { itemStyle: { opacity: 0.95 } },
      markLine: lines.length ? { symbol: 'none', silent: true, data: lines } : undefined,
    }],
  };
}
