// Backend base URL for the live inference API (FastAPI, serve.py).
// Set NEXT_PUBLIC_API_URL in Vercel to your deployed backend, e.g.
//   https://<your-space>.hf.space   or   https://<service>.onrender.com
// Falls back to localhost for `npm run dev` against a local `uvicorn serve:app`.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

export async function predict(file, descriptors) {
  const fd = new FormData();
  fd.append("image", file);
  if (descriptors && Object.keys(descriptors).length) {
    fd.append("descriptors", JSON.stringify(descriptors));
  }
  const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: fd });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j.detail) detail = j.detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}
