import { useState } from "react";
import { validateUpload } from "../lib/api.js";

/** Drag-and-drop or click-to-browse upload. Calls onFile with a valid File. */
export default function ImageDropzone({ onFile, previewUrl, onError }) {
  const [dragging, setDragging] = useState(false);

  const accept = (file) => {
    const problem = validateUpload(file);
    if (problem) {
      onError(problem);
      return;
    }
    onError(null);
    onFile(file);
  };

  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (e.dataTransfer.files[0]) accept(e.dataTransfer.files[0]);
      }}
      className={`flex min-h-[220px] cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed p-6 text-center transition ${
        dragging ? "border-accent bg-accent/10" : "border-white/[0.16] bg-raised hover:border-accent/60"
      }`}
    >
      {previewUrl ? (
        <img src={previewUrl} alt="Uploaded preview" className="max-h-56 rounded-lg object-contain" />
      ) : (
        <>
          <span className="flex h-12 w-12 items-center justify-center rounded-full bg-overlay text-accent-soft">
            <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path
                d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <div>
            <p className="text-sm font-medium text-ink">Drop an image here or click to browse</p>
            <p className="mt-1 text-xs text-ink-faint">JPG or PNG, up to 10 MB</p>
          </div>
        </>
      )}
      <input
        type="file"
        accept="image/png,image/jpeg"
        className="sr-only"
        onChange={(e) => e.target.files[0] && accept(e.target.files[0])}
      />
    </label>
  );
}
