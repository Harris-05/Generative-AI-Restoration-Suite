import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { API_URL, checkHealth } from "../lib/api.js";
import { Chip } from "./Feedback.jsx";

const APP_VERSION = "v0.1.0";

const TABS = [
  { to: "/universal", label: "Universal Restoration" },
  { to: "/hard-routed", label: "Hard-Routed Restoration" },
  { to: "/soft-moe", label: "Soft Mixture-of-Experts" },
  { to: "/face-to-sketch", label: "Face-to-Sketch" },
];

function BackendStatus({ health }) {
  if (health.ok === null) return <Chip>Checking backend</Chip>;
  if (!health.ok) return <Chip tone="danger">Backend: offline</Chip>;
  return <Chip tone="ok">Backend: connected</Chip>;
}

export default function Layout() {
  const [health, setHealth] = useState({ ok: null });

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      const result = await checkHealth();
      if (!cancelled) setHealth(result);
    };
    poll();
    const timer = setInterval(poll, 10000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-white/[0.08] bg-raised/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3 md:px-8">
          <div className="flex items-center gap-3">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-accent to-violet text-white">
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2v3M12 19v3M2 12h3M19 12h3" strokeLinecap="round" />
              </svg>
            </span>
            <span className="text-[0.95rem] font-semibold tracking-tight">Generative AI Restoration Studio</span>
            <Chip tone="accent">{APP_VERSION}-CVLAB</Chip>
          </div>
          <div className="flex items-center gap-2">
            <BackendStatus health={health} />
            <Chip>{health.ok && health.device ? health.device : "device unknown"}</Chip>
          </div>
        </div>
        <nav className="mx-auto max-w-7xl overflow-x-auto px-4 pb-3 md:px-8">
          <ul className="flex gap-1 whitespace-nowrap">
            {TABS.map((tab) => (
              <li key={tab.to}>
                <NavLink
                  to={tab.to}
                  className={({ isActive }) =>
                    `block rounded-lg px-3 py-1.5 text-[0.8125rem] font-medium transition ${
                      isActive ? "bg-overlay text-ink ring-1 ring-accent/40" : "text-ink-muted hover:text-ink"
                    }`
                  }
                >
                  {tab.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 md:px-8 md:py-8">
        <Outlet context={{ health }} />
      </main>

      <footer className="border-t border-white/[0.08]">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2 px-4 py-3 font-mono text-[0.6875rem] text-ink-faint md:px-8">
          <span>Four models served by one FastAPI backend (ONNX Runtime)</span>
          <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer" className="hover:text-ink">
            API docs
          </a>
        </div>
      </footer>
    </div>
  );
}
