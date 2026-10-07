// Monte Carlo Studio — lógica de la interfaz.
import { api, followJob, startPing } from './api.js';
import { t as tr, setLang, getLang, LANGS } from './i18n.js';
import {
  themeColors, prefersReducedMotion, fmtNum, fmtInt, decimalsFor,
  priceOption, errorOption, pathsOption, varOption, histOption, pathsYRange,
} from './charts.js';

// ------------------------------------------------------------------------------------------------
// Utilidades
// ------------------------------------------------------------------------------------------------
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const t = (k, v) => tr(k, v);

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== false && v !== null && v !== undefined) e.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
}

const store = {
  get(k, d) { try { const v = localStorage.getItem('mcs.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('mcs.' + k, JSON.stringify(v)); } catch { /* sin almacenamiento */ } },
};

function toast(msg, err = false) {
  const n = el('div', { class: 'toast' + (err ? ' err' : ''), text: msg });
  $('#toasts').append(n);
  setTimeout(() => { n.style.transition = 'opacity .3s'; n.style.opacity = '0'; setTimeout(() => n.remove(), 320); }, err ? 6500 : 3200);
}

const debounce = (fn, ms) => { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; };

// ------------------------------------------------------------------------------------------------
// Estado
// ------------------------------------------------------------------------------------------------
const S = {
  caps: null,
  model: null, payoff: null,
  method: { family: 'mc', noise: 'raw', variance: 'none', R: 32, n0: 256, M: 2, max_L: 10, n_steps: 0 },
  run: { eps: 0.01, seed: 123, backend: 'cpu', threads: 1, max_seconds: 0 },
  job: null, follower: null, busy: false,
  conv: { pts: [], ref: null, eps: 0.01, refKind: '' },
  levels: null, replicas: null, nSteps: 0,
  lastResult: null,
  results: store.get('results', []),
  paths: { data: null, k: 0, playing: false, speed: 1, seed: 7, dirty: true, loading: false, last: 0 },
};

const charts = {};
let colors = themeColors();
const anim = () => !prefersReducedMotion();

// ------------------------------------------------------------------------------------------------
// Definiciones de la interfaz a partir de /api/capabilities
// ------------------------------------------------------------------------------------------------
const modelDef = (id) => S.caps.models.find((m) => m.id === id);
const payoffDef = (id) => S.caps.payoffs.find((p) => p.id === id);
const allowedPayoffs = (modelId) => (modelId === 'basket' ? ['basket'] : ['european', 'asian', 'geomasian', 'lookback', 'barrier']);

function defaultsOf(def) { const o = {}; def.params.forEach((p) => { o[p.key] = p.default; }); return o; }
function setModel(id) { S.model = { type: id, ...defaultsOf(modelDef(id)) }; }
function setPayoff(id) { S.payoff = { type: id, ...defaultsOf(payoffDef(id)) }; }

function paramLabel(scope, key, ownerType) {
  if (scope === 'payoff' && key === 'sigma') return t('p.sigma_bgk');
  if (scope === 'model' && ownerType === 'basket' && key === 'rho') return t('p.rho_basket');
  return t('p.' + key);
}

// ------------------------------------------------------------------------------------------------
// Campos con deslizador
// ------------------------------------------------------------------------------------------------
function fmtParam(def, v) {
  if (def.step >= 1) return String(Math.round(v));
  const d = Math.max(0, Math.min(4, Math.ceil(-Math.log10(def.step))));
  return Number(v).toFixed(d);
}

function slider({ label, def, value, onInput, log = false, suffix = '' }) {
  const range = el('input', { type: 'range', 'aria-label': label });
  const num = el('input', { type: 'number', step: log ? 'any' : def.step, min: def.min, max: def.max });
  const toPos = (v) => (log ? (Math.log10(v) - Math.log10(def.min)) / (Math.log10(def.max) - Math.log10(def.min)) : (v - def.min) / (def.max - def.min));
  const fromPos = (p) => (log ? Math.pow(10, Math.log10(def.min) + p * (Math.log10(def.max) - Math.log10(def.min))) : def.min + p * (def.max - def.min));
  range.min = 0; range.max = 1000; range.step = 1;
  const setFill = () => range.style.setProperty('--fill', (Number(range.value) / 10) + '%');
  const show = (v) => { range.value = Math.round(Math.max(0, Math.min(1, toPos(v))) * 1000); setFill(); num.value = log ? Number(v).toPrecision(2) : fmtParam(def, v); };
  show(value);
  const snap = (v) => (log ? v : (def.step >= 1 ? Math.round(v) : Math.round(v / def.step) * def.step));
  range.addEventListener('input', () => { const v = snap(fromPos(Number(range.value) / 1000)); num.value = log ? v.toPrecision(2) : fmtParam(def, v); setFill(); onInput(log ? Number(v.toPrecision(2)) : v); });
  num.addEventListener('change', () => { let v = Number(num.value); if (!Number.isFinite(v)) v = value; v = Math.min(def.max, Math.max(def.min, v)); show(v); onInput(v); });
  const row = el('div', { class: 'field' }, el('label', { class: 'lbl' }, el('span', { text: label }), el('span', { text: suffix })), range, num);
  row.setValue = show;
  return row;
}

function buildParamFields(container, scope, ownerType, defs, values, onChange) {
  container.replaceChildren();
  defs.params.forEach((p) => {
    container.append(slider({
      label: paramLabel(scope, p.key, ownerType), def: p, value: values[p.key],
      onInput: (v) => { values[p.key] = v; onChange(); },
    }));
  });
}

// ------------------------------------------------------------------------------------------------
// Formularios: escenario, método y ejecución
// ------------------------------------------------------------------------------------------------
function buildSegment(container, items, current, onPick) {
  container.replaceChildren();
  items.forEach((it) => {
    const b = el('button', { type: 'button', title: it.title || '', disabled: !!it.disabled, class: it.id === current ? 'on' : '' , text: it.label });
    b.addEventListener('click', () => { if (!it.disabled) onPick(it.id); });
    container.append(b);
  });
}

function buildScenario() {
  // presets
  const sp = $('#sel-preset');
  sp.replaceChildren(el('option', { value: '', text: t('preset.custom') }),
    ...S.caps.presets.map((p) => el('option', { value: p.id, text: p.label[getLang()] || p.label.es })));
  sp.value = S.presetId || '';

  buildSegment($('#seg-model'), S.caps.models.map((m) => ({ id: m.id, label: t('model.' + m.id) })), S.model.type, (id) => {
    setModel(id);
    if (!allowedPayoffs(id).includes(S.payoff.type)) setPayoff(allowedPayoffs(id)[0]);
    S.presetId = ''; sanitizeMethod(); buildScenario(); buildMethod(); onScenarioChanged();
  });
  buildParamFields($('#model-params'), 'model', S.model.type, modelDef(S.model.type), S.model, onScenarioChanged);

  const sel = $('#sel-payoff');
  sel.replaceChildren(...allowedPayoffs(S.model.type).map((id) => el('option', { value: id, text: t('payoff.' + id) })));
  sel.value = S.payoff.type;
  buildParamFields($('#payoff-params'), 'payoff', S.payoff.type, payoffDef(S.payoff.type), S.payoff, onScenarioChanged);
  // por defecto el strike sigue a S0 si el usuario no lo ha tocado: se deja tal cual (valores del preset)
}

function variantAllowed(variance) {
  const m = S.model.type, p = S.payoff.type, ml = S.method.family === 'mlmc' || S.method.family === 'mlqmc';
  if (variance === 'none') return true;
  if (variance === 'cv') return (m === 'gbm' && p === 'asian') || (!ml && m === 'dupire' && p === 'european');
  if (variance === 'is') return m === 'gbm' && p === 'european';
  return false;
}

function sanitizeMethod() {
  const M = S.method, m = S.model.type;
  if (m === 'basket' && (M.family === 'mlmc' || M.family === 'mlqmc')) M.family = 'mc';
  if ((m === 'heston' || m === 'basket') && M.noise !== 'raw') M.noise = 'raw';
  if (M.family === 'mc' || M.family === 'mlmc') { /* el ruido solo afecta a QMC */ }
  if (!variantAllowed(M.variance)) M.variance = 'none';
}

function buildMethod() {
  const M = S.method, m = S.model.type;
  const fam = [['mc', 'MC'], ['qmc', 'QMC'], ['mlmc', 'MLMC'], ['mlqmc', 'MLQMC']];
  buildSegment($('#seg-family'), fam.map(([id, label]) => ({
    id, label, disabled: m === 'basket' && id.startsWith('ml'), title: m === 'basket' ? t('warn.nobasketml') : t('fam.' + id),
  })), M.family, (id) => { M.family = id; sanitizeMethod(); buildMethod(); updateWarn(); });
  $('#family-hint').textContent = t('famhint.' + M.family);

  const qmcLike = M.family === 'qmc' || M.family === 'mlqmc';
  $('#wrap-noise').hidden = !qmcLike;
  const noNoise = m === 'heston' || m === 'basket';
  buildSegment($('#seg-noise'), [['raw', t('noise.raw')], ['bb', t('noise.bb')], ['pca', t('noise.pca')]].map(([id, label]) => ({
    id, label, disabled: noNoise && id !== 'raw', title: noNoise && id !== 'raw' ? t('warn.nobbpca') : t('noisehint.' + id),
  })), M.noise, (id) => { M.noise = id; buildMethod(); });

  buildSegment($('#seg-variance'), [['none', t('var.none')], ['cv', t('var.cv')], ['is', t('var.is')]].map(([id, label]) => ({
    id, label, disabled: !variantAllowed(id), title: variantAllowed(id) ? t('varhint.' + id) : t('warn.var_' + id),
  })), M.variance, (id) => { M.variance = id; buildMethod(); });

  // avanzado
  const adv = $('#adv-body');
  adv.replaceChildren(
    slider({ label: t('adv.nsteps'), def: { min: 0, max: 2048, step: 1 }, value: M.n_steps, onInput: (v) => { M.n_steps = v; }, suffix: t('adv.auto0') }),
    ...(qmcLike ? [
      slider({ label: t('adv.R'), def: { min: 2, max: 64, step: 1 }, value: M.R, onInput: (v) => { M.R = v; } }),
      slider({ label: t('adv.n0'), def: { min: 64, max: 4096, step: 64 }, value: M.n0, onInput: (v) => { M.n0 = v; } }),
    ] : []),
    ...(M.family.startsWith('ml') ? [
      slider({ label: t('adv.M'), def: { min: 2, max: 4, step: 1 }, value: M.M, onInput: (v) => { M.M = v; } }),
      slider({ label: t('adv.maxL'), def: { min: 4, max: 12, step: 1 }, value: M.max_L, onInput: (v) => { M.max_L = v; } }),
    ] : []),
  );
  updateWarn();
  updateCoupledToggle();
}

function updateWarn() {
  const box = $('#method-warn');
  const msgs = [];
  const R = S.run;
  if (S.method.family === 'mc' && R.eps < 0.003 && R.backend === 'cpu') msgs.push(t('warn.slow'));
  if (S.method.variance === 'is') msgs.push(t('warn.is'));
  box.hidden = msgs.length === 0;
  box.textContent = msgs.join(' ');
}

function buildRun() {
  const eps = slider({ label: t('lbl.eps'), def: { min: 1e-4, max: 0.1, step: 0 }, value: S.run.eps, log: true,
    onInput: (v) => { S.run.eps = v; updateWarn(); } });
  const f = $('#field-eps');
  f.replaceWith(eps); eps.id = 'field-eps';

  const bk = S.caps.backends;
  const cuda = bk.find((b) => b.id === 'cuda');
  buildSegment($('#seg-backend'), [
    { id: 'cpu', label: 'CPU', title: t('backend.cpu') },
    { id: 'cuda', label: 'GPU (CUDA)', disabled: !cuda.available, title: cuda.available ? t('backend.cuda') : t('backend.nocuda') },
  ], S.run.backend, (id) => { S.run.backend = id; buildRun(); updateChips(); });

  const hw = bk.find((b) => b.id === 'cpu').threads;
  const th = slider({ label: t('lbl.threads'), def: { min: 1, max: hw, step: 1 }, value: Math.min(S.run.threads, hw),
    onInput: (v) => { S.run.threads = v; updateChips(); } });
  th.querySelectorAll('input').forEach((i) => { i.disabled = S.run.backend !== 'cpu'; });
  $('#field-threads').replaceWith(th); th.id = 'field-threads';
}

function buildMaxSec() {
  const sel = $('#sel-maxsec');
  sel.replaceChildren(...[0, 10, 30, 60, 120, 300, 600].map((s) => el('option', { value: s, text: s === 0 ? t('maxsec.none') : `${s} s` })));
  sel.value = String(S.run.max_seconds);
  sel.onchange = () => { S.run.max_seconds = Number(sel.value); };
}

function updateChips() {
  const bk = S.caps.backends;
  const cpu = bk.find((b) => b.id === 'cpu'), cuda = bk.find((b) => b.id === 'cuda');
  const busy = S.busy;
  $('#chip-backend .dot').className = 'dot' + (busy ? ' busy' : '');
  $('#chip-backend-text').textContent = S.run.backend === 'cuda' ? 'GPU (CUDA)' : `CPU · ${cpu.threads} ${t('chip.threadsHW')}`;
  $('#chip-threads').textContent = S.run.backend === 'cuda' ? (cuda.available ? 'CUDA' : t('chip.nogpu')) : `${Math.min(S.run.threads, cpu.threads)} / ${cpu.threads} ${t('chip.threads')}`;
}

function applyPreset(id) {
  const p = S.caps.presets.find((x) => x.id === id);
  if (!p) { S.presetId = ''; return; }
  S.presetId = id;
  S.model = { type: p.model.type, ...defaultsOf(modelDef(p.model.type)), ...p.model };
  S.payoff = { type: p.payoff.type, ...defaultsOf(payoffDef(p.payoff.type)), ...p.payoff };
  Object.assign(S.method, { noise: 'raw', variance: 'none', family: 'mc' }, p.method);
  if (p.method.eps) S.run.eps = p.method.eps;
  delete S.method.eps;
  sanitizeMethod();
  buildScenario(); buildMethod(); buildRun();
  onScenarioChanged();
  toast(t('toast.preset', { name: p.label[getLang()] }));
}

const onScenarioChanged = debounce(() => {
  S.paths.dirty = true;
  if ($('#panel-paths').classList.contains('on')) loadPaths(false);
  updateCoupledToggle();
}, 350);

function updateCoupledToggle() {
  const ml = S.method.family === 'mlmc' || S.method.family === 'mlqmc';
  $('#tgl-coupled').hidden = !ml || S.model.type === 'basket';
  if ($('#tgl-coupled').hidden) $('#chk-coupled').checked = false;
}

// ------------------------------------------------------------------------------------------------
// Ejecución y seguimiento
// ------------------------------------------------------------------------------------------------
function buildRequest() {
  const M = S.method;
  const method = { family: M.family, noise: M.noise, variance: M.variance, eps: S.run.eps, n_steps: M.n_steps };
  if (M.family === 'qmc' || M.family === 'mlqmc') { method.R = M.R; method.n0 = M.n0; }
  if (M.family === 'mlmc' || M.family === 'mlqmc') { method.M = M.M; method.max_L = M.max_L; }
  return {
    backend: S.run.backend, threads: S.run.threads, seed: S.run.seed, max_seconds: S.run.max_seconds,
    model: { ...S.model }, payoff: { ...S.payoff }, method,
  };
}

function resetLive() {
  S.conv = { pts: [], ref: null, eps: S.run.eps, refKind: '' };
  S.levels = null; S.replicas = null; S.nSteps = 0; S.lastResult = null;
  for (const id of ['st-price', 'st-se', 'st-n', 'st-time', 'st-ref']) $('#' + id).textContent = '—';
  for (const id of ['st-price-s', 'st-se-s', 'st-n-s', 'st-time-s', 'st-ref-s']) $('#' + id).textContent = '';
  setProgress(0, '');
  scheduleRedraw();
}

function setBusy(b) {
  S.busy = b;
  $('#btn-run').disabled = b;
  $('#btn-run').classList.toggle('running', b);
  $('#btn-cancel').disabled = !b;
  $('#card-scenario').classList.toggle('refetch', false);
  updateChips();
}

function setProgress(frac, text) {
  $('#progress-bar').style.width = `${Math.round(Math.max(0, Math.min(1, frac)) * 100)}%`;
  $('#status-pct').textContent = frac > 0 && frac < 1 ? `${Math.round(frac * 100)} %` : '';
  if (text !== undefined) $('#status-text').textContent = text;
}

async function startRun() {
  if (S.busy) return;
  resetLive();
  let sub;
  try {
    sub = await api('/api/run', { method: 'POST', body: buildRequest() });
  } catch (e) { toast(e.message, true); return; }
  S.job = sub.job_id;
  setBusy(true);
  setProgress(0, t('status.queued'));
  S.runStart = performance.now();
  switchTab('conv');
  S.follower = followJob(sub.job_id, {
    started: onStarted, progress: onProgress, result: onResult, cancelled: onCancelled, error: onJobError,
  });
}

async function cancelRun() {
  if (!S.busy || !S.job) return;
  $('#btn-cancel').disabled = true;
  try { await api(`/api/jobs/${S.job}/cancel`, { method: 'POST', body: {} }); } catch (e) { toast(e.message, true); }
}

function onStarted(ev) {
  S.threadsUsed = ev.threads;
  if (ev.reference) { S.conv.ref = ev.reference.value; S.conv.refKind = ev.reference.kind; }
  setProgress(0, t('status.planning'));
}

function stageText(ev) {
  switch (ev.stage) {
    case 'plan': return ev.n_steps ? t('status.plan_steps', { n: ev.n_steps }) : t('status.planning');
    case 'pilot': return t('status.pilot');
    case 'main': return t('status.main');
    case 'level': return t('status.level', { L: ev.L });
    case 'doubling': return t('status.doubling', { d: ev.doubling + 1, n: fmtInt(ev.n_per_replica) });
    default: return '';
  }
}

function onProgress(ev) {
  if (ev.n_steps) S.nSteps = ev.n_steps;
  if (ev.reference) { S.conv.ref = ev.reference.value; S.conv.refKind = ev.reference.kind; }
  if (ev.levels) S.levels = ev.levels;
  if (ev.replica_means) S.replicas = ev.replica_means;
  if (ev.n_done > 0 && ev.se > 0 && ev.stage !== 'plan') {
    const last = S.conv.pts[S.conv.pts.length - 1];
    if (!last || ev.n_done > last[0]) S.conv.pts.push([ev.n_done, ev.mean, ev.se]);
    const target = S.run.eps / Math.SQRT2;
    const frac = Math.min(0.995, Math.pow(target / ev.se, 2));
    setProgress(Math.max(frac, 0.02), stageText(ev));
    liveStats(ev);
  } else {
    setProgress(0, stageText(ev));
  }
  scheduleRedraw();
}

function liveStats(ev) {
  const dec = decimalsFor(ev.se);
  $('#st-price').textContent = fmtNum(ev.mean, dec);
  $('#st-price-s').textContent = `± ${fmtNum(1.96 * ev.se, dec)} (95 %)`;
  $('#st-se').textContent = ev.se.toExponential(2);
  $('#st-se-s').textContent = `ε/√2 = ${(S.run.eps / Math.SQRT2).toExponential(2)}`;
  $('#st-n').textContent = fmtInt(ev.n_done);
  $('#st-n-s').textContent = ev.n_done.toLocaleString('en-US').replace(/,/g, ' ');
  $('#st-time').textContent = `${ev.elapsed.toFixed(2)} s`;
  $('#st-time-s').textContent = ev.elapsed > 0 ? `${(ev.n_done / ev.elapsed / 1e6).toFixed(2)} M ${t('unit.paths_s')}` : '';
}

function countUp(node, to, fmt, ms = 520) {
  if (!anim() || !Number.isFinite(to)) { node.textContent = fmt(to); return; }
  const from = Number(node.dataset.v) || to * 0.8;
  const t0 = performance.now();
  const step = (now) => {
    const p = Math.min(1, (now - t0) / ms);
    const e = 1 - Math.pow(1 - p, 3);
    node.textContent = fmt(from + (to - from) * e);
    if (p < 1) requestAnimationFrame(step);
  };
  node.dataset.v = to;
  requestAnimationFrame(step);
}

function finishCommon() {
  setBusy(false);
  if (S.follower) { S.follower.close(); S.follower = null; }
}

function onResult(r) {
  finishCommon();
  S.lastResult = r;
  if (r.reference) { S.conv.ref = r.reference.value; S.conv.refKind = r.reference.kind; }
  const dec = decimalsFor(r.std_error);
  countUp($('#st-price'), r.price, (v) => fmtNum(v, dec));
  $('#st-price-s').textContent = `95 %: [${fmtNum(r.ci95[0], dec)}, ${fmtNum(r.ci95[1], dec)}]`;
  $('#st-se').textContent = r.std_error.toExponential(2);
  $('#st-se-s').textContent = r.std_error <= S.run.eps / Math.SQRT2 * 1.0001 ? t('stat.se_ok') : t('stat.se_notmet');
  $('#st-n').textContent = fmtInt(r.n_samples);
  $('#st-n-s').textContent = r.n_samples.toLocaleString('en-US').replace(/,/g, ' ');
  $('#st-time').textContent = `${r.time_s.toFixed(2)} s`;
  $('#st-time-s').textContent = `${r.mpaths_per_s.toFixed(2)} M ${t('unit.paths_s')} · ${r.backend === 'cuda' ? 'GPU' : r.threads + ' ' + t('chip.threads')}`;
  const refNode = $('#st-ref'), refS = $('#st-ref-s');
  if (r.reference) {
    refNode.textContent = fmtNum(r.reference.value, dec);
    refS.replaceChildren(el('span', { class: 'badge ' + (r.within ? 'ok' : 'no'), text: `${r.within ? '✓' : '✕'} |Δ| = ${r.abs_error.toExponential(1)}` }),
      el('span', { class: 'muted', text: `  z = ${r.z_score.toFixed(1)}` }));
    refNode.title = r.reference.kind;
  } else {
    refNode.textContent = '—';
    refS.textContent = t('stat.noref');
  }
  const bits = [];
  if (r.truncated) bits.push(t('status.truncated'));
  if (r.n_nonfinite) bits.push(t('status.nonfinite', { n: r.n_nonfinite }));
  if (r.beta !== undefined) bits.push(`β = ${r.beta.toFixed(3)}`);
  if (r.z_star !== undefined) bits.push(`z* = ${r.z_star.toFixed(2)}`);
  if (r.n_steps) bits.push(`${r.n_steps} ${t('unit.steps')}`);
  setProgress(1, `${t('status.done')}${bits.length ? ' · ' + bits.join(' · ') : ''}`);
  if (r.levels && r.levels.length) S.levels = r.levels;
  const last = S.conv.pts[S.conv.pts.length - 1];
  if (!last || r.n_samples > last[0]) S.conv.pts.push([r.n_samples, r.price, r.std_error]);
  addResult(r);
  scheduleRedraw();
  if (r.truncated) toast(t('toast.truncated'), false);
}

function onCancelled(ev) {
  finishCommon();
  setProgress(0, t('status.cancelled'));
  if (ev.partial) onPartial(ev.partial);
  toast(t('toast.cancelled'));
}

function onPartial(r) {
  if (!r || !Number.isFinite(r.price)) return;
  const dec = decimalsFor(r.std_error);
  $('#st-price').textContent = fmtNum(r.price, dec);
  $('#st-price-s').textContent = t('status.partial');
}

function onJobError(ev) {
  finishCommon();
  setProgress(0, t('status.error'));
  toast(ev.message || t('status.error'), true);
}

// ------------------------------------------------------------------------------------------------
// Gráficos de convergencia
// ------------------------------------------------------------------------------------------------
let redrawPending = false;
function scheduleRedraw() {
  if (redrawPending) return;
  redrawPending = true;
  requestAnimationFrame(() => { redrawPending = false; drawConvergence(); });
}

function drawConvergence() {
  const d = S.conv;
  const has = d.pts.length > 0;
  $('#empty-price').hidden = has; $('#empty-error').hidden = has;
  if (!has) {
    $('#empty-price').innerHTML = `<div><b>${t('empty.title')}</b>${t('empty.text')}</div>`;
    $('#empty-error').innerHTML = `<div><b>${t('empty.title2')}</b>${t('empty.text2')}</div>`;
  }
  if (!has) {
    // Sin datos: lienzo vacío (los ejes sin sentido solo estorbarían al mensaje)
    charts.price.setOption({ animation: false, xAxis: { show: false }, yAxis: { show: false }, series: [], legend: { show: false }, toolbox: { show: false }, dataZoom: [] }, { notMerge: true });
    charts.error.setOption({ animation: false, xAxis: { show: false }, yAxis: { show: false }, series: [], legend: { show: false }, toolbox: { show: false } }, { notMerge: true });
    drawExtra();
    return;
  }
  charts.price.setOption(priceOption(d, colors, anim(), t), { notMerge: true });
  charts.error.setOption(errorOption(d, colors, anim(), t), { notMerge: true });
  drawExtra();
}

// Paneles adicionales de convergencia: niveles de MLMC o réplicas de QMC (tablas compactas)
function drawExtra() {
  const box = $('#conv-extra');
  if (S.levels && S.levels.length && (S.method.family === 'mlmc' || S.method.family === 'mlqmc')) {
    const rows = S.levels.map((l) => el('tr', {}, el('td', { text: l.l }), el('td', { text: fmtInt(l.N) }),
      el('td', { text: l.E.toExponential(2) }), el('td', { text: l.V.toExponential(2) })));
    box.replaceChildren(el('div', { class: 'hint', text: t('conv.levels') }),
      el('div', { class: 'table-wrap' }, el('table', { class: 'results', style: 'min-width:0' },
        el('thead', {}, el('tr', {}, ...['l', 'N', 'E[ΔP]', 'Var[ΔP]'].map((h) => el('th', { text: h })))), el('tbody', {}, ...rows))));
  } else {
    box.replaceChildren();
  }
}

// ------------------------------------------------------------------------------------------------
// Trayectorias
// ------------------------------------------------------------------------------------------------
function pathsBody() {
  const coupled = $('#chk-coupled').checked && !$('#tgl-coupled').hidden;
  const body = {
    model: { ...S.model }, payoff: { ...S.payoff },
    n_steps: Number($('#sel-steps').value), n_paths: Number($('#in-npaths').value), seed: S.paths.seed,
    sobol: S.paths.source === 'sobol', construction: S.paths.construction || 'raw', fan: 2000,
  };
  if (coupled) {
    const lvl = Math.round(Math.log(body.n_steps) / Math.log(S.method.M || 2));
    body.coupled = { level: Math.max(1, Math.min(lvl, 9)), M: S.method.M || 2 };
  }
  return body;
}

async function loadPaths(newSeed) {
  if (S.paths.loading) { S.paths.again = true; return; }
  if (newSeed) S.paths.seed = Math.floor(Math.random() * 1e6);
  S.paths.loading = true;
  $('#panel-paths .chart-card').classList.add('refetch');
  try {
    const data = await api('/api/paths', { method: 'POST', body: pathsBody() });
    S.paths.data = data;
    S.paths.dirty = false;
    S.paths.k = 0;
    S.paths.fan = data.fan || null;
    S.paths.yr = pathsYRange(data, data.fan, data.marks);
    $('#card-var').hidden = !data.paths[0].V;
    if (data.paths[0].V) charts.var.resize();   // se creó con el panel oculto (ancho 0)
    $('#paths-title').textContent = `${t('paths.title')} — ${t('model.' + data.model_type)} · ${t('payoff.' + data.payoff_type)}`;
    $('#paths-note').textContent = noteFor(data);
    startPlay(true);
  } catch (e) {
    toast(e.message, true);
  } finally {
    S.paths.loading = false;
    $$('#panel-paths .chart-card').forEach((c) => c.classList.remove('refetch'));
    if (S.paths.again) { S.paths.again = false; loadPaths(false); }
  }
}

function noteFor(data) {
  const parts = [];
  if (data.coupled) parts.push(t('paths.note_coupled', { f: data.coupled.t_fine.length - 1, c: data.coupled.t_coarse.length - 1, pf: data.coupled.payoff_fine, pc: data.coupled.payoff_coarse }));
  else parts.push(t('paths.note', { n: data.paths.length, m: data.t.length - 1 }));
  if (data.fan) parts.push(t('paths.note_fan', { m: data.fan.payoff_mean }));
  return parts.join(' ');
}

function pathView() {
  const d = S.paths.data;
  const coupled = !!d.coupled && $('#chk-coupled').checked;
  return {
    data: d, k: Math.floor(S.paths.k), fan: S.paths.fan, showFan: $('#chk-fan').checked, showMean: $('#chk-mean').checked,
    showRun: $('#chk-run').checked, coupled, mu: S.model.mu, T: S.model.T, S0: S.model.S0 ?? S.model.S0, yr: S.paths.yr,
  };
}

function drawPaths(final = false) {
  if (!S.paths.data) return;
  const v = pathView();
  charts.paths.setOption(pathsOption(v, colors, false, t), { notMerge: true, lazyUpdate: true });
  if (S.paths.data.paths[0].V) charts.var.setOption(varOption(v, colors, false, t), { notMerge: true, lazyUpdate: true });
  if (final || !S.paths.histDrawn) { charts.hist.setOption(histOption(v, colors, anim(), t), { notMerge: true }); S.paths.histDrawn = true; $('#empty-hist').hidden = !!(S.paths.fan); }
}

function startPlay(restart) {
  if (!S.paths.data) return;
  if (restart || S.paths.k >= S.paths.data.t.length - 1) S.paths.k = 0;
  S.paths.histDrawn = false;
  S.paths.playing = true;
  S.paths.last = performance.now();
  if (!anim()) { S.paths.k = S.paths.data.t.length - 1; S.paths.playing = false; }
  updatePlayBtn();
  drawPaths(true);
  if (S.paths.playing) requestAnimationFrame(playTick);
}

function playTick(now) {
  if (!S.paths.playing) return;
  const n = S.paths.data.t.length - 1;
  const dt = Math.min(0.1, (now - S.paths.last) / 1000);
  S.paths.last = now;
  S.paths.k = Math.min(n, S.paths.k + dt * (n / 3.2) * S.paths.speed);
  drawPaths(false);
  if (S.paths.k >= n) { S.paths.playing = false; updatePlayBtn(); drawPaths(true); return; }
  requestAnimationFrame(playTick);
}

function updatePlayBtn() {
  const n = S.paths.data ? S.paths.data.t.length - 1 : 0;
  const done = S.paths.k >= n && n > 0;
  $('#btn-play').textContent = S.paths.playing ? t('paths.pause') : (done ? t('paths.replay') : t('paths.play'));
}

function togglePlay() {
  if (!S.paths.data) { loadPaths(false); return; }
  if (S.paths.playing) { S.paths.playing = false; updatePlayBtn(); return; }
  const n = S.paths.data.t.length - 1;
  if (S.paths.k >= n) S.paths.k = 0;
  S.paths.playing = true; S.paths.last = performance.now(); updatePlayBtn();
  requestAnimationFrame(playTick);
}

function buildPathControls() {
  const steps = $('#sel-steps');
  steps.replaceChildren(...[32, 64, 128, 256, 512].map((n) => el('option', { value: n, text: String(n) })));
  steps.value = '128';
  steps.onchange = () => loadPaths(false);

  const rg = $('#rg-npaths'), inp = $('#in-npaths');
  const setFill = () => rg.style.setProperty('--fill', ((rg.value - rg.min) / (rg.max - rg.min) * 100) + '%');
  setFill();
  const reload = debounce(() => loadPaths(false), 250);
  rg.oninput = () => { inp.value = rg.value; setFill(); reload(); };
  inp.onchange = () => { inp.value = Math.max(1, Math.min(200, Number(inp.value) || 5)); rg.value = inp.value; setFill(); reload(); };

  S.paths.source = 'pseudo'; S.paths.construction = 'raw';
  const drawSeg = () => {
    buildSegment($('#seg-source'), [{ id: 'pseudo', label: t('paths.pseudo') }, { id: 'sobol', label: 'Sobol' }], S.paths.source,
      (id) => { S.paths.source = id; drawSeg(); loadPaths(false); });
    const noBB = S.model.type === 'heston' || S.model.type === 'basket';
    buildSegment($('#seg-construction'), [{ id: 'raw', label: 'Raw' }, { id: 'bb', label: 'BB', disabled: noBB }, { id: 'pca', label: 'PCA', disabled: noBB }],
      noBB ? 'raw' : S.paths.construction, (id) => { S.paths.construction = id; drawSeg(); loadPaths(false); });
    if (noBB) S.paths.construction = 'raw';
  };
  S.drawPathSegs = drawSeg;
  drawSeg();

  $('#btn-redraw').onclick = () => loadPaths(true);
  $('#btn-play').onclick = togglePlay;
  const sp = $('#rg-speed');
  const spFill = () => sp.style.setProperty('--fill', ((sp.value - sp.min) / (sp.max - sp.min) * 100) + '%');
  sp.oninput = () => { S.paths.speed = Math.pow(2, Number(sp.value)); $('#speed-label').textContent = `${S.paths.speed >= 1 ? S.paths.speed.toFixed(S.paths.speed % 1 ? 1 : 0) : S.paths.speed.toFixed(2)}×`; spFill(); };
  spFill();
  for (const id of ['chk-fan', 'chk-mean', 'chk-run']) $('#' + id).onchange = () => drawPaths(true);
  $('#chk-coupled').onchange = () => loadPaths(false);
}

// ------------------------------------------------------------------------------------------------
// Resultados
// ------------------------------------------------------------------------------------------------
function methodLabel(r) {
  let s = r.family;
  if (['QMC', 'MLQMC'].includes(r.family)) s += ' ' + ({ raw: 'Raw', bb: 'BB', pca: 'PCA' }[r.noise] || r.noise);
  if (r.variance === 'cv') s += ' + CV';
  if (r.variance === 'is') s += ' + IS';
  return s;
}

function addResult(r) {
  const entry = {
    id: (S.results[0]?.id || 0) + 1, time: new Date().toLocaleTimeString(),
    method: methodLabel(r), scenario: `${t('model.' + S.model.type)} · ${t('payoff.' + S.payoff.type)}`,
    price: r.price, se: r.std_error, n: r.n_samples, secs: r.time_s, mps: r.mpaths_per_s,
    err: r.reference ? r.abs_error : null, ok: r.reference ? r.within : null, eps: r.eps,
    backend: r.backend === 'cuda' ? 'GPU' : `CPU×${r.threads}`,
  };
  S.results.unshift(entry);
  S.results = S.results.slice(0, 100);
  store.set('results', S.results);
  S.freshId = entry.id;
  renderResults();
}

function resultRows() {
  return S.results.map((r) => [r.id, r.time, r.method, r.scenario, r.backend, r.eps, r.price, r.se, r.n, r.secs, r.mps, r.err, r.ok]);
}

function renderResults() {
  const tbl = $('#tbl-results');
  const head = ['#', t('res.time'), t('res.method'), t('res.scenario'), t('res.backend'), 'ε', t('res.price'), t('res.se'), 'N', t('res.secs'), 'M paths/s', '|Δ|', 'OK'];
  const dec = (r) => decimalsFor(r.se);
  tbl.replaceChildren(
    el('thead', {}, el('tr', {}, ...head.map((h) => el('th', { text: h })))),
    el('tbody', {}, ...S.results.map((r) => el('tr', { class: r.id === S.freshId ? 'fresh' : '' },
      el('td', { text: r.id }), el('td', { text: r.time }), el('td', { text: r.method }), el('td', { text: r.scenario }), el('td', { text: r.backend }),
      el('td', { text: r.eps }), el('td', { text: fmtNum(r.price, dec(r)) }), el('td', { text: r.se.toExponential(2) }), el('td', { text: r.n.toLocaleString('en-US').replace(/,/g, ' ') }),
      el('td', { text: r.secs.toFixed(2) }), el('td', { text: r.mps.toFixed(2) }), el('td', { text: r.err === null ? '—' : r.err.toExponential(1) }),
      el('td', {}, r.ok === null ? el('span', { class: 'badge neutral', text: '—' }) : el('span', { class: 'badge ' + (r.ok ? 'ok' : 'no'), text: r.ok ? '✓' : '✕' }))))),
  );
  $('#res-empty').textContent = S.results.length ? '' : t('res.empty');
  $('#res-empty').hidden = S.results.length > 0;
  tbl.parentElement.hidden = S.results.length === 0;
}

function resultsText(sep) {
  const head = ['#', 'time', 'method', 'scenario', 'backend', 'eps', 'price', 'se', 'N', 'seconds', 'Mpaths_s', 'abs_err', 'ok'];
  const rows = resultRows().map((r) => r.map((v) => (v === null ? '' : typeof v === 'boolean' ? (v ? 'yes' : 'no') : v)));
  if (sep === 'md') {
    return ['| ' + head.join(' | ') + ' |', '|' + head.map(() => '---').join('|') + '|', ...rows.map((r) => '| ' + r.join(' | ') + ' |')].join('\n');
  }
  const esc = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
  return [head.join(','), ...rows.map((r) => r.map(esc).join(','))].join('\n');
}

// ------------------------------------------------------------------------------------------------
// Pestañas, tema, idioma
// ------------------------------------------------------------------------------------------------
function switchTab(id) {
  $$('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === id));
  $$('.tab-panel').forEach((p) => p.classList.toggle('on', p.id === 'panel-' + id));
  requestAnimationFrame(() => {
    Object.values(charts).forEach((c) => c.resize());
    if (id === 'paths' && (S.paths.dirty || !S.paths.data)) loadPaths(false);
  });
}

function applyTheme(mode) {
  if (mode === 'light' || mode === 'dark') document.documentElement.dataset.theme = mode;
  else delete document.documentElement.dataset.theme;
  store.set('theme', mode);
  requestAnimationFrame(() => {
    colors = themeColors();
    drawConvergence();
    if (S.paths.data) { S.paths.histDrawn = false; drawPaths(true); }
  });
}

function currentThemeIsDark() {
  const attr = document.documentElement.dataset.theme;
  return attr ? attr === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function applyI18n() {
  $$('[data-i18n]').forEach((n) => { n.textContent = t(n.dataset.i18n); });
  $$('[data-i18n-title]').forEach((n) => { n.title = t(n.dataset.i18nTitle); });
  document.documentElement.lang = getLang();
  $('#btn-lang').textContent = getLang().toUpperCase();
  document.title = 'Monte Carlo Studio';
}

function rebuildAll() {
  applyI18n();
  buildScenario(); buildMethod(); buildRun(); buildMaxSec(); updateChips();
  S.drawPathSegs && S.drawPathSegs();
  renderResults(); updatePlayBtn(); drawConvergence();
  if (S.paths.data) { $('#paths-note').textContent = noteFor(S.paths.data); S.paths.histDrawn = false; drawPaths(true); }
}

// ------------------------------------------------------------------------------------------------
// Arranque
// ------------------------------------------------------------------------------------------------
async function boot() {
  setLang(store.get('lang', (navigator.language || 'es').startsWith('en') ? 'en' : 'es'));
  applyTheme(store.get('theme', 'auto'));
  try {
    S.caps = await api('/api/capabilities');
  } catch (e) {
    document.body.innerHTML = `<div style="padding:40px;font-family:system-ui;max-width:560px;margin:auto"><h2>Monte Carlo Studio</h2><p>${e.message}</p><p>${t('boot.token')}</p></div>`;
    return;
  }
  const cpu = S.caps.backends.find((b) => b.id === 'cpu');
  S.run.threads = cpu.threads;
  S.run.backend = 'cpu';
  setModel('gbm'); setPayoff('european');
  S.presetId = '';

  charts.price = echarts.init($('#chart-price'), null, { renderer: 'canvas' });
  charts.error = echarts.init($('#chart-error'), null, { renderer: 'canvas' });
  charts.paths = echarts.init($('#chart-paths'), null, { renderer: 'canvas' });
  charts.var = echarts.init($('#chart-var'), null, { renderer: 'canvas' });
  charts.hist = echarts.init($('#chart-hist'), null, { renderer: 'canvas' });
  new ResizeObserver(() => Object.values(charts).forEach((c) => c.resize())).observe(document.body);

  buildPathControls();
  rebuildAll();

  $('#sel-preset').addEventListener('change', (e) => applyPreset(e.target.value));
  $('#sel-payoff').addEventListener('change', (e) => { setPayoff(e.target.value); S.presetId = ''; sanitizeMethod(); buildScenario(); buildMethod(); onScenarioChanged(); });
  $('#in-seed').addEventListener('change', (e) => { S.run.seed = Math.max(0, Math.floor(Number(e.target.value) || 0)); });
  $('#btn-dice').addEventListener('click', () => { S.run.seed = Math.floor(Math.random() * 1e6); $('#in-seed').value = S.run.seed; });
  $('#btn-run').addEventListener('click', startRun);
  $('#btn-cancel').addEventListener('click', cancelRun);
  $$('.tabs button').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
  $('#btn-theme').addEventListener('click', () => applyTheme(currentThemeIsDark() ? 'light' : 'dark'));
  $('#btn-lang').addEventListener('click', () => { setLang(LANGS[(LANGS.indexOf(getLang()) + 1) % LANGS.length]); store.set('lang', getLang()); rebuildAll(); });
  $('#btn-quit').addEventListener('click', async () => {
    if (!confirm(t('quit.confirm'))) return;
    try { await api('/api/shutdown', { method: 'POST', body: {} }); } catch { /* ya cerrado */ }
    document.body.innerHTML = `<div style="padding:60px;font-family:system-ui;text-align:center"><h2>${t('quit.done')}</h2></div>`;
    setTimeout(() => window.close(), 400);
  });
  $('#btn-copy-md').addEventListener('click', () => copyText(resultsText('md')));
  $('#btn-copy-csv').addEventListener('click', () => copyText(resultsText('csv')));
  $('#btn-dl-csv').addEventListener('click', () => {
    const a = el('a', { href: URL.createObjectURL(new Blob([resultsText('csv')], { type: 'text/csv' })), download: 'monte-carlo-resultados.csv' });
    document.body.append(a); a.click(); a.remove();
  });
  $('#btn-clear').addEventListener('click', () => { S.results = []; store.set('results', []); renderResults(); });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (!document.documentElement.dataset.theme) applyTheme('auto'); });
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); startRun(); }
    if (e.key === 'Escape') cancelRun();
  });

  startPing();
  window.__mcs = { S, startRun, loadPaths, drawPaths, charts };   // utilidades de depuración desde la consola
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast(t('toast.copied')); }
  catch { toast(t('toast.copyfail'), true); }
}

boot();
