// Cliente de la API local. El token viaja en el fragmento de la URL (#token=...), que el navegador
// nunca envía al servidor; aquí se lee una vez y se guarda en sessionStorage.

let token = '';
try {
  const h = new URLSearchParams(location.hash.slice(1));
  token = h.get('token') || sessionStorage.getItem('mc_token') || '';
  if (token) sessionStorage.setItem('mc_token', token);
} catch { /* sessionStorage puede no estar disponible */ }
if (location.hash) history.replaceState(null, '', location.pathname);

export async function api(path, { method = 'GET', body } = {}) {
  const res = await fetch(path, {
    method,
    headers: { 'X-MC-Token': token, ...(body ? { 'Content-Type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch { /* sin cuerpo JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
  return data;
}

// Sigue un trabajo por SSE; si el navegador no puede (o la conexión falla), cae a sondeo largo.
// handlers: { started, progress, result, cancelled, error } — cada uno recibe el objeto del evento.
export function followJob(jobId, handlers) {
  let closed = false;
  let es = null;
  let lastSeq = 0;
  let polling = false;

  const dispatch = (type, data) => {
    if (data.seq && data.seq <= lastSeq) return;           // duplicados al reconectar
    if (data.seq) lastSeq = data.seq;
    const h = handlers[type];
    if (h) h(data);
    if (type === 'result' || type === 'cancelled' || type === 'error') close();
  };

  const close = () => { closed = true; if (es) { es.close(); es = null; } };

  const poll = async () => {
    if (polling) return;
    polling = true;
    while (!closed) {
      try {
        const r = await api(`/api/jobs/${jobId}/poll?after=${lastSeq}&wait=1500`);
        for (const ev of r.events) dispatch(ev.type, ev);
        if (r.finished) break;
      } catch (e) {
        if (!closed) { handlers.error && handlers.error({ message: e.message }); }
        break;
      }
    }
    polling = false;
  };

  if (typeof EventSource === 'undefined') { poll(); }
  else {
    es = new EventSource(`/api/jobs/${jobId}/events?token=${encodeURIComponent(token)}`);
    for (const t of ['started', 'progress', 'result', 'cancelled', 'error'])
      es.addEventListener(t, (e) => { try { dispatch(t, JSON.parse(e.data)); } catch { /* evento mal formado */ } });
    es.onerror = () => { if (!closed && es && es.readyState === 2) { es.close(); es = null; poll(); } };
  }
  return { close };
}

// El servidor se apaga solo si la ventana deja de hacer ping (ver ServerOptions::watchdog).
export function startPing() {
  const ping = () => fetch('/api/ping', { headers: { 'X-MC-Token': token } }).catch(() => {});
  ping();
  setInterval(ping, 4000);
}
