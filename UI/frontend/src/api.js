const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message = typeof payload === "object" ? payload.error : payload;
    throw new Error(message || `請求失敗 (${response.status})`);
  }
  return payload;
}

export function videoUrl(path) {
  return `${API_BASE}${path}`;
}
