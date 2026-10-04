# Restoration Studio (frontend)

React + Tailwind frontend for the four generative-AI workspaces. The design follows the
Stitch "Obsidian Lumina" system exported to `Frontend/Stitch Photos/`.

## Run it

```bash
npm install
npm run dev          # http://localhost:5173
```

The app expects the FastAPI backend at `http://localhost:8000`. To point it elsewhere, set
`VITE_API_URL` in a `.env` file. Without a running backend the header shows "Backend: offline"
and the run buttons report that the backend is unavailable.

## Pages

| Route | Workspace |
|---|---|
| `/universal` | Universal Restoration: upload a corrupted image, or pick a clean sample and apply a corruption |
| `/hard-routed` | Hard-Routed Restoration: classifier probabilities, predicted corruption, selected expert |
| `/soft-moe` | Soft Mixture-of-Experts: routing weights with the dominant expert marked |
| `/face-to-sketch` | Face-to-Sketch: upload or webcam, Style 1/2/3, download the sketch |

## Backend contract

`src/lib/api.js` lists the endpoints the backend must implement: `/health`, `/samples`,
`/universal-restoration`, `/hard-routing`, `/soft-mixture`, and `/face-to-sketch`.

## Build

```bash
npm run build        # outputs dist/
```
