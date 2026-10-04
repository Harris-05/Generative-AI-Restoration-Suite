const TONES = {
  neutral: "border-white/[0.08] bg-overlay text-ink-muted",
  accent: "border-accent/40 bg-accent/10 text-accent-soft",
  ok: "border-ok/30 bg-ok/10 text-ok-soft",
  danger: "border-danger/40 bg-danger/10 text-danger",
};

export function Chip({ children, tone = "neutral", className = "" }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-[0.6875rem] tracking-wide tabular ${TONES[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function InferencePill({ ms }) {
  if (ms === undefined || ms === null) return null;
  return <Chip tone="ok">Inference {ms.toFixed(0)} ms</Chip>;
}

export function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div role="alert" className="rounded-lg border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
      {message}
    </div>
  );
}

export function Spinner({ label = "Running model..." }) {
  return (
    <div className="flex items-center gap-3 text-sm text-ink-muted">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
      {label}
    </div>
  );
}

export function EmptyState({ title = "Upload an image to get started", text, children }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-white/[0.08] bg-raised p-10 text-center">
      <span className="flex h-12 w-12 items-center justify-center rounded-full bg-overlay text-accent-soft">
        <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.8">
          <rect x="4" y="4" width="16" height="16" rx="3" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      </span>
      <p className="text-base font-semibold">{title}</p>
      {text && <p className="max-w-md text-sm text-ink-muted">{text}</p>}
      {children && <div className="mt-2 flex flex-wrap justify-center gap-2">{children}</div>}
    </div>
  );
}

export function Card({ title, aside, children, className = "" }) {
  return (
    <section className={`rounded-xl border border-white/[0.08] bg-raised p-5 ${className}`}>
      {(title || aside) && (
        <div className="mb-4 flex items-center justify-between gap-3">
          {title && <h2 className="font-mono text-[0.6875rem] uppercase tracking-wider text-ink-muted">{title}</h2>}
          {aside}
        </div>
      )}
      {children}
    </section>
  );
}

export function PageHeader({ title, chip, description, aside }) {
  return (
    <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
      <div>
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold tracking-tight md:text-[1.75rem]">{title}</h1>
          {chip}
        </div>
        <p className="mt-2 max-w-3xl text-sm text-ink-muted">{description}</p>
      </div>
      {aside && <div className="flex flex-wrap items-center gap-2">{aside}</div>}
    </div>
  );
}

/** Donut chart drawn with a CSS conic gradient. items: [{ value (0..1), color (hex) }] */
export function Donut({ items, center }) {
  let acc = 0;
  const stops = items
    .map((item) => {
      const start = acc;
      acc += item.value * 100;
      return `${item.color} ${start}% ${acc}%`;
    })
    .join(", ");
  return (
    <div className="relative mx-auto h-44 w-44 rounded-full" style={{ background: `conic-gradient(${stops})` }}>
      <div className="absolute inset-6 flex flex-col items-center justify-center rounded-full bg-raised">
        {center}
      </div>
    </div>
  );
}
