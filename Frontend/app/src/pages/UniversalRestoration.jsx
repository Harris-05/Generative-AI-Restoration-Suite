import { useEffect, useState } from "react";
import { useOutletContext } from "react-router-dom";
import ImageDropzone from "../components/ImageDropzone.jsx";
import ImagePanel from "../components/ImagePanel.jsx";
import { Card, Chip, EmptyState, ErrorBanner, InferencePill, PageHeader, Spinner } from "../components/Feedback.jsx";
import { CORRUPTIONS, getSamples, imageSrc, postForm } from "../lib/api.js";

const SEVERITIES = ["low", "medium", "high"];

function Metric({ label, value, unit }) {
  return (
    <div className="rounded-lg border border-white/[0.08] bg-overlay px-4 py-3">
      <p className="font-mono text-[0.6875rem] uppercase tracking-wider text-ink-muted">{label}</p>
      <p className="mt-1 font-mono text-lg tabular">
        {value}
        {unit && <span className="ml-1 text-xs text-ink-muted">{unit}</span>}
      </p>
    </div>
  );
}

export default function UniversalRestoration() {
  const { health } = useOutletContext();
  const [source, setSource] = useState("upload");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [samples, setSamples] = useState([]);
  const [samplesError, setSamplesError] = useState(null);
  const [sampleId, setSampleId] = useState(null);
  const [corruption, setCorruption] = useState("salt_pepper");
  const [severity, setSeverity] = useState("high");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getSamples()
      .then(setSamples)
      .catch(() => setSamplesError("Samples load from the backend. Start the API to see them."));
  }, []);

  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const canRun = source === "upload" ? Boolean(file) : Boolean(sampleId);
  const corruptionInfo = CORRUPTIONS[corruption];

  const run = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res =
        source === "upload"
          ? await postForm("/universal-restoration", { image: file })
          : await postForm("/universal-restoration", { sample_id: sampleId, corruption, severity });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Universal Degradation Restoration"
        chip={<Chip tone="accent">Universal autoencoder</Chip>}
        description="One autoencoder restores salt-and-pepper noise, blur, or occlusion without being told which corruption is present."
        aside={
          <>
            <Chip tone={health.ok ? "ok" : "danger"}>{health.ok ? "Backend: connected" : "Backend: offline"}</Chip>
            {result?.inference_ms !== undefined && <InferencePill ms={result.inference_ms} />}
            {result?.psnr !== undefined && <Chip tone="accent">PSNR {result.psnr.toFixed(1)} dB</Chip>}
            {result?.ssim !== undefined && <Chip tone="accent">SSIM {result.ssim.toFixed(3)}</Chip>}
          </>
        }
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
        <div className="flex flex-col gap-4">
          <Card title="Data ingestion">
            <div className="mb-4 inline-flex rounded-lg border border-white/[0.08] bg-overlay p-1 text-sm">
              {[
                ["upload", "Upload corrupted image"],
                ["sample", "Pick a clean sample"],
              ].map(([value, text]) => (
                <button
                  key={value}
                  onClick={() => {
                    setSource(value);
                    setResult(null);
                  }}
                  className={`rounded-md px-3 py-1.5 font-medium transition ${
                    source === value ? "bg-accent text-white" : "text-ink-muted hover:text-ink"
                  }`}
                >
                  {text}
                </button>
              ))}
            </div>

            {source === "upload" ? (
              <ImageDropzone
                previewUrl={previewUrl}
                onFile={(f) => {
                  setFile(f);
                  setResult(null);
                }}
                onError={setError}
              />
            ) : (
              <>
                <p className="mb-3 font-mono text-[0.6875rem] uppercase tracking-wider text-ink-muted">
                  Benchmark calibration datasets
                </p>
                {samplesError && <p className="text-sm text-ink-faint">{samplesError}</p>}
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {samples.map((s) => (
                    <button key={s.id} onClick={() => setSampleId(s.id)} className="group text-left">
                      <div
                        className={`relative aspect-square overflow-hidden rounded-lg border-2 transition ${
                          sampleId === s.id ? "border-accent" : "border-transparent group-hover:border-white/20"
                        }`}
                      >
                        <img src={s.url} alt={s.label} className="h-full w-full object-cover" />
                        {sampleId === s.id && (
                          <span className="absolute right-1.5 top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-accent text-white">
                            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="3">
                              <path d="M5 12l5 5 9-10" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          </span>
                        )}
                      </div>
                      <span className="mt-1 block truncate font-mono text-[0.6875rem] text-ink-muted">{s.label}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </Card>

          {source === "sample" && (
            <Card title="Corruption synthesis">
              <label className="mb-4 block text-sm">
                <span className="mb-1.5 block text-ink-muted">Corruption type</span>
                <select
                  value={corruption}
                  onChange={(e) => setCorruption(e.target.value)}
                  className="w-full rounded-lg border border-white/[0.12] bg-overlay px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
                >
                  {Object.entries(CORRUPTIONS).map(([key, c]) => (
                    <option key={key} value={key}>
                      {c.label}
                    </option>
                  ))}
                </select>
              </label>
              <div className="mb-4">
                <span className="mb-1.5 block text-sm text-ink-muted">Severity</span>
                <div className="grid grid-cols-3 gap-1 rounded-lg border border-white/[0.08] bg-overlay p-1 text-sm">
                  {SEVERITIES.map((s) => (
                    <button
                      key={s}
                      onClick={() => setSeverity(s)}
                      className={`rounded-md px-3 py-1.5 capitalize transition ${
                        severity === s ? "bg-accent text-white" : "text-ink-muted hover:text-ink"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
              <p className="rounded-lg border border-white/[0.08] bg-overlay px-3 py-2 font-mono text-[0.75rem] text-ink-muted">
                Active configuration: {corruptionInfo.label}, {severity} ({corruptionInfo.levels[severity]})
              </p>
            </Card>
          )}

          <button
            onClick={run}
            disabled={!canRun || loading}
            className="rounded-lg bg-accent px-4 py-3 text-sm font-semibold text-white transition hover:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? "Restoring..." : "Run restoration"}
          </button>
          <ErrorBanner message={error} />
        </div>

        <div className="flex flex-col gap-4">
          <Card title="Comparative tensor inspection" aside={<Chip>1:1 pixel space</Chip>}>
            {loading && <Spinner />}
            {!loading && !result && (
              <EmptyState
                title="Upload an image to get started"
                text="Choose a corrupted image, or pick a clean sample and apply a corruption to it."
              >
                <button
                  onClick={() => setSource("upload")}
                  className="rounded-lg border border-white/[0.12] bg-overlay px-3 py-2 text-sm text-ink hover:border-accent"
                >
                  Select local file
                </button>
                <button
                  onClick={() => setSource("sample")}
                  className="rounded-lg border border-white/[0.12] bg-overlay px-3 py-2 text-sm text-ink hover:border-accent"
                >
                  Load sample
                </button>
              </EmptyState>
            )}
            {result && (
              <div className="grid gap-4 md:grid-cols-3">
                <ImagePanel
                  src={imageSrc(result.input_png)}
                  caption="Input (corrupted)"
                  tone="input"
                  tag={source === "sample" ? `${severity} severity` : undefined}
                />
                <ImagePanel src={imageSrc(result.restored_png)} caption="Restored output" tone="output" tag="Universal AE" />
                {result.original_png && (
                  <ImagePanel src={imageSrc(result.original_png)} caption="Original (clean)" tone="reference" tag="Reference" />
                )}
              </div>
            )}
          </Card>

          {result && (
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Metric label="Inference" value={result.inference_ms.toFixed(0)} unit="ms" />
              {result.psnr !== undefined && <Metric label="PSNR" value={result.psnr.toFixed(2)} unit="dB" />}
              {result.ssim !== undefined && <Metric label="SSIM" value={result.ssim.toFixed(3)} />}
              <Metric label="Device" value={health.device || "unknown"} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
