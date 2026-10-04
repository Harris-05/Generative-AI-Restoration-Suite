// Backend client. The endpoints below are the contract the FastAPI backend must implement.
//   GET  /health                -> { status: "ok", device: "cuda:0" | "cpu", softmax_temperature: 0.85 }
//   GET  /samples               -> [{ id, label, url }]   (clean reference images with ground truth)
//   POST /universal-restoration -> form: image  OR  sample_id + corruption + severity
//                                  returns: input_png, restored_png, original_png?, psnr?, ssim?, inference_ms
//   POST /hard-routing          -> form: image
//                                  returns: probabilities {clean, salt_pepper, blur, occlusion},
//                                           predicted, expert, input_png, restored_png, inference_ms
//   POST /soft-mixture          -> form: image
//                                  returns: weights {clean, salt_pepper, blur, occlusion}, dominant,
//                                           input_png, restored_png, inference_ms
//   POST /face-to-sketch        -> form: image, style (0, 1 or 2)
//                                  returns: sketch_png, inference_ms
// PSNR and SSIM are returned only when a clean reference exists (samples). Uploads have none.

export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png"];

/** Returns an error message for an unacceptable file, or null if it is fine. */
export function validateUpload(file) {
  if (!ALLOWED_TYPES.includes(file.type)) return "Unsupported file type. Use JPG or PNG.";
  if (file.size > MAX_UPLOAD_BYTES) return "File too large (max 10 MB).";
  return null;
}

export function imageSrc(base64) {
  return base64 ? `data:image/png;base64,${base64}` : null;
}

/** Returns { ok, device, softmax_temperature }. ok is false when the backend cannot be reached. */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_URL}/health`);
    if (!res.ok) return { ok: false };
    const body = await res.json();
    return { ok: true, device: body.device, softmax_temperature: body.softmax_temperature };
  } catch {
    return { ok: false };
  }
}

export async function getSamples() {
  const res = await fetch(`${API_URL}/samples`);
  if (!res.ok) throw new Error("Could not load sample images.");
  return res.json();
}

/** POSTs multipart form data and returns the JSON body. Throws with the backend's message. */
export async function postForm(path, fields = {}) {
  const form = new FormData();
  Object.entries(fields).forEach(([key, value]) => {
    if (value !== undefined && value !== null) form.append(key, value);
  });
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, { method: "POST", body: form });
  } catch {
    throw new Error("Backend is offline. Start the API and try again.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status}).`);
  }
  return res.json();
}

/** Keys and labels used by every routing screen, in the order the assignment lists them. */
export const BRANCHES = [
  { key: "clean", label: "Clean (identity)", color: "#94a3b8", tw: "bg-ink-muted" },
  { key: "salt_pepper", label: "Salt-and-pepper expert", color: "#6366f1", tw: "bg-accent" },
  { key: "blur", label: "Blur expert", color: "#a855f7", tw: "bg-violet" },
  { key: "occlusion", label: "Occlusion expert", color: "#06b6d4", tw: "bg-cyan" },
];

/** Test-grid severities from the assignment (the same values the evaluation uses). */
export const CORRUPTIONS = {
  salt_pepper: {
    label: "Salt-and-pepper (impulse noise)",
    levels: { low: "p = 3%", medium: "p = 8%", high: "p = 15%" },
  },
  blur: {
    label: "Gaussian blur",
    levels: { low: "kernel 3, σ = 0.7", medium: "kernel 5, σ = 1.5", high: "kernel 7, σ = 2.5" },
  },
  occlusion: {
    label: "Rectangular occlusion",
    levels: { low: "1 box, ~10% area", medium: "2 boxes, ~20% area", high: "3 boxes, ~35% area" },
  },
};
