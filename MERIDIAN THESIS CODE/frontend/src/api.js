const BASE = 'http://localhost:8000';

async function get(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
}

export const api = {
  health:    ()       => get('/health'),
  shortlist: ()       => get('/shortlist'),
  stock:     (ticker) => get(`/stock/${ticker}`),
};
