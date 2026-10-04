import { useEffect, useRef, useState } from "react";
import { useOutletContext } from "react-router-dom";
import ImageDropzone from "../components/ImageDropzone.jsx";
import ImagePanel from "../components/ImagePanel.jsx";
import { Card, Chip, EmptyState, ErrorBanner, InferencePill, PageHeader, Spinner } from "../components/Feedback.jsx";
import { imageSrc, postForm } from "../lib/api.js";

const STYLES = [
  { value: 0, label: "Style 1" },
  { value: 1, label: "Style 2" },
  { value: 2, label: "Style 3" },
];

/** Live webcam preview with the capture button overlaid. Returns the frame as a File. */
function WebcamCapture({ onCapture, onError }) {
  const videoRef = useRef(null);
  const [streaming, setStreaming] = useState(false);

  useEffect(() => {
    let stream;
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((s) => {
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = s;
        setStreaming(true);
      })
      .catch(() => onError("Camera unavailable. Allow camera access or upload a photo instead."));
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, [onError]);

  const capture = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) onCapture(new File([blob], "webcam-capture.png", { type: "image/png" }));
    }, "image/png");
  };

  return (
    <div className="relative overflow-hidden rounded-xl border border-white/[0.08] bg-raised">
      <video ref={videoRef} autoPlay playsInline muted className="aspect-video w-full object-cover" />
      <div className="absolute left-3 top-3">
        <Chip tone="danger">● Live webcam</Chip>
      </div>
      <div className="absolute inset-x-0 bottom-3 flex justify-center">
        <button
          onClick={capture}
          disabled={!streaming}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white shadow-lg transition hover:bg-accent-soft disabled:opacity-40"
        >
          Capture snapshot
        </button>
      </div>
    </div>
  );
}

export default function FaceToSketch() {
  useOutletContext();
  const [mode, setMode] = useState("upload");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [style, setStyle] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const run = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await postForm("/face-to-sketch", { image: file, style }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const download = () => {
    const link = document.createElement("a");
    link.href = imageSrc(result.sketch_png);
    link.download = `sketch-style-${style + 1}.png`;
    link.click();
  };

  return (
    <div>
      <PageHeader
        title="Face-to-Sketch Generator"
        chip={<Chip tone="accent">Conditional cGAN</Chip>}
        description="Turn a face photo into a sketch in one of three styles. Upload a photo, or capture one with your webcam."
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
        <div className="flex flex-col gap-4">
          <Card title="Acquisition">
            <div className="mb-4 inline-flex rounded-lg border border-white/[0.08] bg-overlay p-1 text-sm">
              {[
                ["upload", "Upload photo"],
                ["webcam", "Use webcam"],
              ].map(([value, text]) => (
                <button
                  key={value}
                  onClick={() => {
                    setMode(value);
                    setFile(null);
                    setResult(null);
                  }}
                  className={`rounded-md px-3 py-1.5 font-medium transition ${
                    mode === value ? "bg-accent text-white" : "text-ink-muted hover:text-ink"
                  }`}
                >
                  {text}
                </button>
              ))}
            </div>
            {mode === "upload" ? (
              <ImageDropzone
                previewUrl={previewUrl}
                onFile={(f) => {
                  setFile(f);
                  setResult(null);
                }}
                onError={setError}
              />
            ) : (
              <WebcamCapture
                onCapture={(f) => {
                  setFile(f);
                  setResult(null);
                }}
                onError={setError}
              />
            )}
          </Card>

          <Card title="Synthesis style">
            <div className="grid grid-cols-3 gap-3">
              {STYLES.map((s) => (
                <button
                  key={s.value}
                  onClick={() => setStyle(s.value)}
                  className={`relative rounded-lg border px-3 py-4 text-left text-sm transition ${
                    style === s.value
                      ? "border-accent bg-accent/10 text-ink"
                      : "border-white/[0.08] bg-overlay text-ink-muted hover:text-ink"
                  }`}
                >
                  <span className="block font-mono text-[0.6875rem] text-ink-faint">0{s.value + 1}</span>
                  <span className="mt-1 block font-medium">{s.label}</span>
                </button>
              ))}
            </div>
          </Card>

          <button
            onClick={run}
            disabled={!file || loading}
            className="rounded-lg bg-accent px-4 py-3 text-sm font-semibold text-white transition hover:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? "Generating..." : "Generate sketch"}
          </button>
          <ErrorBanner message={error} />
        </div>

        <div className="flex flex-col gap-4">
          <Card title="Synthesis canvas" aside={<Chip>Side by side</Chip>}>
            {loading && <Spinner />}
            {!loading && !result && (
              <EmptyState title="Upload an image to get started" text="Upload or capture a photo, pick a style, then generate its sketch." />
            )}
            {result && (
              <div className="grid grid-cols-2 gap-4">
                <ImagePanel src={previewUrl} caption="Original photo" tone="input" />
                <ImagePanel src={imageSrc(result.sketch_png)} caption="Generated sketch" tone="output" tag={`Style ${style + 1}`} />
              </div>
            )}
          </Card>

          {result && (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <InferencePill ms={result.inference_ms} />
              <button
                onClick={download}
                className="rounded-lg border border-white/[0.12] bg-overlay px-4 py-2 text-sm font-medium text-ink transition hover:border-accent"
              >
                Download sketch
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
