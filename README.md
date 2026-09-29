# SatQuery ARC

**Search the archive by meaning. Verify every change.**

SatQuery ARC is a semantic search and verified change-detection workspace for a
satellite-imagery archive. You describe what you are looking for in plain
language ("newly built-up expansion near a river", "reservoir filling"), the
system ranks matching sites from the indexed archive, and for any site it runs a
transparent, fully offline change-verification pipeline that returns a
*calibrated* verdict — not just a heatmap.

> Smart India Hackathon problem statement **SIH26227**.

Every number shown in the UI is measured from the pixels. Language models, when
configured, only help phrase questions and explanations; they never produce a
measurement. The verification path runs entirely offline.

---

## What it does

- **Semantic archive search.** Natural-language queries are parsed into
  land-cover concepts (built-up, water, vegetation, bare) and modifiers (change,
  proximity to water), then scored against each area of interest's measured
  spectral fractions. Results carry a match score, a one-line reason, and tags.
- **Verified multi-temporal change.** For a selected site, the pipeline compares
  two acquisitions, computes the change (CCDC-style delta + change-vector
  analysis over spectral indices), runs a **confounder firewall** (seasonality,
  cloud/haze, misregistration, radiometric drift), and returns a verdict —
  *supported*, *refuted*, or *inconclusive* — with a confidence breakdown, an
  observation timeline, and the earliest supported date.
- **Analyst review queue + audit log.** Verified candidates land in a queue an
  analyst confirms or rejects. Every decision is appended to an immutable,
  attributable audit log and the queue is re-ranked by confidence.
- **Provenance & export.** Each result exposes its source scenes, models, index
  engine, and a checksum, and can be exported as GeoJSON or an evidence record.

## Tech stack

| Layer | Stack |
| --- | --- |
| Backend | Python 3.12, FastAPI, rasterio/GDAL, NumPy, scikit-image, Pydantic |
| Frontend | React 19, TypeScript, Vite 7, Tailwind CSS v4, react-leaflet, framer-motion |
| Imagery | Sentinel-2 L2A (optical), Sentinel-1 (SAR), Landsat — real GeoTIFFs on disk |
| Embedder | `spectral-concept-v1` (offline, NumPy); optional RemoteCLIP hook if weights are present |

## Repository layout

```
satquery/
├─ backend/            FastAPI service
│  ├─ app/
│  │  ├─ api/          HTTP routes (search, verify, review, archive, health, ...)
│  │  ├─ core/         archive index, embedding, search, verify, review, orchestrator
│  │  ├─ models/       Pydantic schemas
│  │  └─ tools/        measurement engines (change, confounders, indices, verdict, ...)
│  ├─ data/
│  │  ├─ samples/            curated sample scenes (committed)
│  │  └─ cache/aoi_probe/    AOI-probe imagery the archive is built from (committed)
│  ├─ tests/           pytest suite
│  └─ requirements.txt
├─ frontend/           React + Vite app
│  └─ src/
│     ├─ pages/        Landing, ArcStudio (Search & Review), Studio (classic)
│     ├─ components/   ArcMap, panels, account menu, health, ...
│     └─ lib/          API client + hooks
└─ mock/               static HTML mock of the target design
```

## Getting started

### Prerequisites

- Python 3.12+
- Node.js 20+ (22 recommended) and npm

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # macOS / Linux
pip install -r requirements.txt

# Configuration is optional — the app runs fully offline without any keys.
copy .env.example .env                 # then fill in keys only if you want them

# Run the API (builds the archive index on first request)
python -m uvicorn app.main:app --reload --port 8000
```

The API serves on `http://127.0.0.1:8000` (interactive docs at `/docs`).

No credentials are required to run the demo: with `LLM_PROVIDER=none` (or no keys
at all) the offline rule router and the deterministic measurement engines do the
whole job. Optional keys (Mistral / OpenRouter / Gemini for narration, Copernicus
/ AWS for live SAR) are documented in `backend/.env.example`.

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173/`, then **Open the Studio** (or go straight to
`http://localhost:5173/studio`). The dev server proxies `/api` to the backend on
port 8000.

## Using the workspace

1. The Studio opens on a featured query and auto-verifies the top result.
2. Type a query in the header search box (e.g. *"newly built-up urban
   expansion"*, *"reservoir filling and new water bodies"*).
3. Select any result — on the map or in the list — to run verification.
4. Confirm (**C**) or reject (**R**) candidates in the review queue; decisions are
   written to the audit log.
5. Export a verified change as GeoJSON or an evidence record from the Provenance
   panel.

## API overview

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/search` | Semantic / similarity search over the archive |
| `POST` | `/api/verify/{aoi_key}` | Run the change → confounder → verdict pipeline for a site |
| `GET`  | `/api/review` | The analyst review queue + audit log |
| `POST` | `/api/review/{item_id}/decision` | Confirm / reject / reopen a candidate |
| `GET`  | `/api/audit` | The audit log |
| `GET`  | `/api/archive` | Archive overview (tiles, AOIs, stats) |
| `GET`  | `/api/health` | Component health + active capabilities |

## Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q      # backend suite

cd ..\frontend
npm run build                                 # typecheck + production build
npm test                                      # vitest
```

## Notes on data and honesty

- The committed `backend/data/samples` and `backend/data/cache/aoi_probe` are the
  real GeoTIFFs the offline archive is built from. Session uploads, the generated
  archive index, and the review-queue state are runtime artifacts and are not
  committed.
- The active embedder reports as `spectral-concept-v1`. A RemoteCLIP hook
  activates automatically only if the weights are present locally; otherwise the
  provenance shows the models actually in use rather than aspirational names.
- Verdicts are real. A site whose change cannot be confirmed from the available
  bands returns *inconclusive* by design, rather than a manufactured number.
