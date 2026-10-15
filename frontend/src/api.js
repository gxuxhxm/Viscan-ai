const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = err.detail || detail;
    } catch {
      // Non-JSON error body; keep the generic HTTP status detail
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health:       ()        => apiFetch('/health'),
  listPatients: ()        => apiFetch('/patients'),
  getReports:   (pid)     => apiFetch(`/patients/${pid}/reports`),
  getReport:    (rid)     => apiFetch(`/reports/${rid}`),
  query:        (body)    => apiFetch('/query',    { method: 'POST', body: JSON.stringify(body) }),
  dictation:    (body)    => apiFetch('/dictation',{ method: 'POST', body: JSON.stringify(body) }),
  followUp:     (body)    => apiFetch('/follow-up',{ method: 'POST', body: JSON.stringify(body) }),
  saveReport:   (body)    => apiFetch('/reports',   { method: 'POST', body: JSON.stringify(body) }),
};
