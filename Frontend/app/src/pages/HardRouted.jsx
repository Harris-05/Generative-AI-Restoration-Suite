import { useEffect, useState } from "react";
import { useOutletContext } from "react-router-dom";
import ImageDropzone from "../components/ImageDropzone.jsx";
import ImagePanel from "../components/ImagePanel.jsx";
import { Card, Chip, EmptyState, ErrorBanner, InferencePill, PageHeader, Spinner } from "../components/Feedback.jsx";
import { BRANCHES, imageSrc, postForm } from "../lib/api.js";

const CLASSES = BRANCHES;

function labelOf(key) {
  return CLASSES.find((c) => c.key === key)?.label ?? key;
}

function Tile({ label, children }) {
  return (
    <div className="rounded-lg border border-white/[0.08] bg-overlay px-4 py-3">
      <p className="font-mono text-[0.6875rem] uppercase tracking-wider text-ink-muted">{label}</p>
      <div className="mt-1 text-sm font-semibold">{children}</div>
    </div>
  );
}

export default function HardRouted() {
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
      setResult(await postForm("/hard-routing", { image: file }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const top = result ? Math.max(...CLASSES.map((c) => result.probabilities[c.key])) : 0;

  return (
    <div>
      <PageHeader
        title="Hard-Routed Multi-Model Restoration"
        chip={<Chip tone="accent">Benchmark routing pipeline</Chip>}
        description="A classifier picks one corruption class and sends the image to one specialist. Clean images bypass the specialists entirely."
        aside={
          <>
            <Tile label="Model topology">1 classifier + 3 specialists</Tile>
            <Tile label="Routing policy">Hard argmax</Tile>
          </>
        }
      />

      <Card className="mb-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex-1">
            <ImageDropzone previewUrl={previewUrl} onFile={setFile} onError={setError} />
          </div>
          <div className="flex w-full flex-col gap-3 md:w-72">
            <div className="rounded-lg border border-white/[0.08] bg-overlay px-4 py-3">
              <p className="font-mono text-[0.6875rem] uppercase tracking-wider text-ink-muted">Device target</p>
              <p className="mt-1 font-mono text-sm">{health.ok ? health.device || "unknown" : "backend offline"}</p>
            </div>
            <button
              onClick={run}
              disabled={!file || loading}
              className="rounded-lg bg-accent px-4 py-3 text-sm font-semibold text-white transition hover:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "Routing..." : "Run hard-routed restoration"}
            </button>
          </div>
        </div>
      </Card>

      <ErrorBanner message={error} />
      {loading && <Spinner />}

      {!loading && !result && (
        <EmptyState title="Upload an image to get started" text="The classifier names the corruption, then the matching specialist restores the image." />
      )}

      {result && (
        <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
          <div className="flex flex-col gap-4">
            <Card title="Classifier probabilities" aside={<Chip>Softmax layer</Chip>}>
              <ul className="flex flex-col gap-3">
                {CLASSES.map((c) => {
                  const p = result.probabilities[c.key];
                  const isSelected = c.key === result.predicted;
                  return (
                    <li
                      key={c.key}
                      className={`rounded-lg px-3 py-2.5 ${isSelected ? "border border-accent/40 bg-accent/10" : ""}`}
                    >
                      <div className="mb-2 flex items-center justify-between text-sm">
                        <span className={isSelected ? "font-semibold" : "text-ink-muted"}>{c.label}</span>
                        <span className="flex items-center gap-2">
                          {isSelected && <Chip tone="accent">Selected backend</Chip>}
                          <span className="font-mono tabular text-ink-muted">{(p * 100).toFixed(1)}%</span>
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-overlay">
                        <div className={`h-full rounded-full ${isSelected ? "bg-accent" : "bg-accent/30"}`} style={{ width: `${p * 100}%` }} />
                      </div>
                    </li>
                  );
                })}
              </ul>
            </Card>

            <Card title="Dispatch summary">
              <div className="grid grid-cols-2 gap-3">
                <Tile label="Predicted corruption">{labelOf(result.predicted)}</Tile>
                <Tile label="Selected expert">{labelOf(result.expert)}</Tile>
                <Tile label="Classifier confidence">
                  <span className="font-mono tabular">{(top * 100).toFixed(1)}%</span>
                </Tile>
                <Tile label="Inference">
                  <InferencePill ms={result.inference_ms} />
                </Tile>
              </div>
            </Card>
          </div>

          <div className="flex flex-col gap-4">
            <Card
              title="Comparative analysis"
              aside={
                <a
                  href={imageSrc(result.restored_png)}
                  download="restored.png"
                  className="rounded-lg border border-white/[0.12] bg-overlay px-3 py-1.5 text-xs font-medium text-ink hover:border-accent"
                >
                  Download PNG
                </a>
              }
            >
              <div className="grid grid-cols-2 gap-4">
                <ImagePanel src={previewUrl} caption="Input (corrupted)" tone="input" />
                <ImagePanel src={imageSrc(result.restored_png)} caption="Restored output" tone="output" tag={labelOf(result.expert)} />
              </div>
            </Card>

            <Card title="Hard gate routing trace">
              <p className="font-mono text-xs leading-6 text-ink-muted">
                Input → Classifier → [
                {CLASSES.map((c, i) => (
                  <span key={c.key}>
                    {i > 0 && ", "}
                    {result.probabilities[c.key].toFixed(3)}
                  </span>
                ))}
                ] → argmax → <span className="text-accent-soft">{labelOf(result.expert)}</span>
              </p>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
