const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Production builds read the JSON snapshot written by scripts/snapshot_api.py
// instead of a live API (query params are ignored). Deliberately not keyed on
// REACT_APP_API_URL: .env.local sets it for dev and CRA loads that in builds too.
const STATIC = process.env.NODE_ENV === "production";

async function get(path, params) {
  const url = STATIC
    ? new URL(`${process.env.PUBLIC_URL}/api${path}.json`, window.location.origin)
    : new URL(API_BASE + path);
  if (params && !STATIC) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null) url.searchParams.set(k, v);
    });
  }
  // no-cache = always revalidate (a cheap 304 when unchanged). The host's
  // default max-age otherwise lets a browser keep serving the pre-qualifying
  // snapshot after a redeploy, and a page refresh doesn't clear it because
  // these fetches happen on click, not on load.
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  upcoming: () => get("/races/upcoming"),
  upcomingDriverFeatures: (driverId) => get(`/races/upcoming/${driverId}/features`),
  trackRecord: (n) => get("/track-record", n ? { n } : undefined),
  recentRaces: (n) => get("/races/recent", n ? { n } : undefined),
  shareCard: (raceId) => get(`/races/${raceId}/share-card`),
};
