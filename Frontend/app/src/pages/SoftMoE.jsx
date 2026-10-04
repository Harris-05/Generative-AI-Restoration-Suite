import { useEffect, useState } from "react";
import { useOutletContext } from "react-router-dom";
import ImageDropzone from "../components/ImageDropzone.jsx";
import ImagePanel from "../components/ImagePanel.jsx";
import { Card, Chip, Donut, EmptyState, ErrorBanner, InferencePill, PageHeader, Spinner } from "../components/Feedback.jsx";
import { BRANCHES, imageSrc, postForm } from "../lib/api.js";

export default function SoftMoE() {
  const { health } = useOutletContext();
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
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
      setResult(await postForm("/soft-mixture", { image: file }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const weights = result ? result.weights : null;
  const sum = weights ? BRANCHES.reduce((s, b) => s + weights[b.key], 0) : 0;
  const entropy = weights
    ? -BRANCHES.reduce((s, b) => {
        const w = weights[b.key];
        return w > 0 ? s + w * Math.log(w) : s;
      }, 0)
    : 0;
  const dominantLabel = result ? BRANCHES.find((b) => b.key === result.dominant)?.label : null;

  return (
    <div>
      <PageHeader
        title="Soft Mixture-of-Experts Restoration"
        chip={<Chip tone="accent">Softmax gating</Chip>}
        description="A gating network blends the identity branch and all three specialists with continuous weights. The weights show how much each branch contributed."
      />

      <Card className="mb-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex-1">
            <ImageDropzone previewUrl={previewUrl} onFile={setFile} onError={setError} />
          </div>
          <div className="flex w-full flex-col gap-3 md:w-72">
            <p className="font-mono text-xs text-ink-muted">
              {health.ok
                ? `Ready for inference · softmax T = ${health.softmax_temperature ?? "n/a"}`
                : "Backend offline. Start the API to run inference."}
            </p>
            <button
              onClick={run}
              disabled={!file || loading}
              className="rounded-lg bg-accent px-4 py-3 text-sm font-semibold text-white transition hover:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "Blending experts..." : "Run mixture"}
            </button>
          </div>
        </div>
      </Card>

      <ErrorBanner message={error} />
      {loading && <Spinner />}

      {!loading && !result && (
        <EmptyState title="Upload an image to get started" text="Feed a degraded image in to see how the gating network allocates experts." />
      )}

      {result && (
        <>
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
            <Card
              title="Routing weights (gating network)"
              aside={<Chip>Softmax ensemble</Chip>}
            >
              <ul className="flex flex-col gap-3">
                {BRANCHES.map((b) => {
                  const w = weights[b.key];
                  const isDominant = b.key === result.dominant;
                  return (
                    <li
                      key={b.key}
                      className={`rounded-lg px-3 py-2.5 ${isDominant ? "border border-accent/40 bg-accent/10" : ""}`}
                    >
                      <div className="mb-2 flex items-center justify-between text-sm">
                        <span className="flex items-center gap-2">
                          <span className={`h-2.5 w-2.5 rounded-full ${b.tw}`} />
                          {b.label}
                          {isDominant && <Chip tone="accent">Dominant expert</Chip>}
                        </span>
                        <span className="font-mono tabular text-ink-muted">
                          {w.toFixed(2)} ({(w * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-overlay">
                        <div className={`h-full rounded-full ${b.tw}`} style={{ width: `${w * 100}%` }} />
                      </div>
                    </li>
                  );
                })}
              </ul>
              <div className="mt-4 flex flex-wrap justify-between gap-2 font-mono text-[0.6875rem] text-ink-faint">
                <span>Σ w = {sum.toFixed(3)}</span>
                <span>Entropy: {entropy.toFixed(3)} nats</span>
              </div>
            </Card>

            <Card title="Weight distribution" aside={<Chip>Soft split</Chip>}>
              <Donut
                items={BRANCHES.map((b) => ({ value: weights[b.key], color: b.color }))}
                center={
                  <>
                    <span className="text-lg font-semibold">4 experts</span>
                    <span className="font-mono text-[0.625rem] uppercase tracking-wider text-ink-muted">active</span>
                  </>
                }
              />
              <ul className="mt-5 grid grid-cols-2 gap-2 text-xs">
                {BRANCHES.map((b) => (
                  <li key={b.key} className="flex items-center gap-2 text-ink-muted">
                    <span className={`h-2 w-2 rounded-full ${b.tw}`} />
                    {b.label} ({(weights[b.key] * 100).toFixed(0)}%)
                  </li>
                ))}
              </ul>
            </Card>
          </div>

          <Card title="Tensor comparison" aside={<InferencePill ms={result.inference_ms} />} className="mt-6">
            <div className="grid grid-cols-2 gap-4">
              <ImagePanel src={previewUrl} caption="Input (degraded)" tone="input" />
              <ImagePanel src={imageSrc(result.restored_png)} caption="Restored output (weighted)" tone="output" tag={dominantLabel} />
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
