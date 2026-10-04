import { Chip } from "./Feedback.jsx";

const DOT = {
  input: "bg-danger",
  output: "bg-ok",
  reference: "bg-accent-soft",
};

/** One image with a caption row. Shows a placeholder until the image exists. */
export default function ImagePanel({ src, caption, tone = "input", tag, alt }) {
  return (
    <figure className="flex min-w-0 flex-col gap-2">
      <figcaption className="flex items-center justify-between gap-2 text-[0.6875rem] font-mono uppercase tracking-wider text-ink-muted">
        <span className="flex min-w-0 items-center gap-2">
          <span className={`h-2 w-2 shrink-0 rounded-full ${DOT[tone]}`} />
          <span className="truncate">{caption}</span>
        </span>
        {tag && <Chip tone="accent">{tag}</Chip>}
      </figcaption>
      <div className="relative flex aspect-square items-center justify-center overflow-hidden rounded-xl border border-white/[0.08] bg-raised">
        {src ? (
          <img src={src} alt={alt || caption} className="h-full w-full object-contain" />
        ) : (
          <span className="text-xs text-ink-faint">No image yet</span>
        )}
      </div>
    </figure>
  );
}
