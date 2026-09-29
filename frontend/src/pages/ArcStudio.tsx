import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Wordmark } from "../components/Wordmark";
import { ArcMap } from "../components/arc/ArcMap";
import {
  ChangeAnalysisPanel,
  ProvenancePanel,
  ReviewQueuePanel,
  SearchResultsPanel,
} from "../components/arc/panels";
import { arc, type SearchResponse, type SearchResult } from "../lib/arc";
import { useReview, useSearch, useVerification } from "../lib/useArc";
import { useHealth } from "../lib/useHealth";
import { useAuth } from "../lib/auth";
import { AccountMenu } from "../components/arc/AccountMenu";
import { PanelBoundary } from "../components/PanelBoundary";

/**
 * SatQuery ARC - Search & Review workspace.
 *
 * Three columns, mission-control dark: the semantic search results on the left,
 * the archive map over the verified-change analysis in the centre, and the
 * analyst review queue over provenance on the right. Everything here is driven
 * by the live backend; there is no mock data.
 *
 * The centre and right columns are height-managed on xl so nothing scrolls the
 * page on a presentation display; below xl the columns stack and the page
 * scrolls normally.
 */

// A water-oriented default so the dashboard opens on a settled, supported
// verdict ("Confirmed change"), which is the strongest first impression. The
// built-up story is one query away in the search box.
const FEATURED_QUERY = "reservoir filling and new water bodies";

// ---------------------------------------------------------------------------
// Header pieces
// ---------------------------------------------------------------------------

function SearchIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="7" cy="7" r="5" stroke="currentColor" strokeWidth="1.5" />
      <path d="M11 11l3.5 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function Spinner() {
  return (
    <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-signal/30 border-t-signal" />
  );
}

/** Live "everything is local" indicator, backed by the real health probe. */
function OfflinePill() {
  const { error, loading } = useHealth();
  const dot = error ? "bg-disagree" : loading ? "bg-ink-faint" : "bg-agree";
  const label = error ? "Backend unreachable" : "All models local \u00b7 offline";
  return (
    <div
      className="flex items-center gap-2 rounded-full border border-edge bg-hull/70 px-3 py-1.5"
      title={
        error
          ? "The SatQuery ARC backend is not answering."
          : "Embedder, change engine and verdict run on this machine. No data leaves the building."
      }
    >
      <span className="relative flex h-2 w-2">
        <span className={`absolute inset-0 rounded-full ${dot}`} />
        {!error && !loading && (
          <span className="absolute inset-0 animate-[var(--animate-pulse-ring)] rounded-full bg-agree/70" />
        )}
      </span>
      <span className="hidden whitespace-nowrap text-[11px] text-ink-dim sm:inline">{label}</span>
    </div>
  );
}

/** A short, real facet summary of the current results, shown beside the query. */
function facetSummary(res: SearchResponse | null): string {
  if (!res || res.results.length === 0) return "";
  const regions = Array.from(new Set(res.results.map((r) => r.region).filter(Boolean)));
  const sensors = Array.from(new Set(res.results.map((r) => r.sensor).filter(Boolean)));
  const years = res.results
    .map((r) => r.acquisition_date)
    .filter((d): d is string => Boolean(d))
    .map((d) => d.slice(0, 4));
  const parts: string[] = [];
  if (regions.length) {
    parts.push(`AOI: ${regions.slice(0, 2).join(", ")}${regions.length > 2 ? "\u2026" : ""}`);
  }
  if (years.length) {
    const min = years.reduce((a, b) => (a < b ? a : b));
    const max = years.reduce((a, b) => (a > b ? a : b));
    parts.push(min === max ? min : `${min} \u2013 ${max}`);
  }
  if (sensors.length) parts.push(sensors.slice(0, 3).join(" \u00b7 "));
  return parts.join("   \u00b7   ");
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function ArcStudio() {
  const auth = useAuth();
  const search = useSearch();
  const verif = useVerification();
  const review = useReview();

  // The workspace expects a profile in the corner; a reviewer who reached it
  // without signing in still gets one. This is the browser-only demo identity.
  useEffect(() => {
    if (!auth.signedIn) auth.signIn();
  }, [auth]);

  const [queryText, setQueryText] = useState(FEATURED_QUERY);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  const { run, response, busy: searching } = search;
  const { verify, verification, busy: verifying, error: verifyError, aoiKey: verifyingKey } = verif;
  const { state: reviewState, refresh: refreshReview, decide } = review;

  // Select an AOI: show its verified change, then pull the (now-seeded) queue.
  const selectAoi = useCallback(
    async (aoiKey: string) => {
      setSelectedKey(aoiKey);
      const v = await verify(aoiKey);
      if (v) await refreshReview();
    },
    [verify, refreshReview],
  );

  // Featured search on first load, then auto-verify the top usable result so the
  // dashboard opens fully populated rather than on an empty shell.
  const didInit = useRef(false);
  useEffect(() => {
    if (didInit.current) return;
    didInit.current = true;
    void (async () => {
      const res = await run({ text: FEATURED_QUERY, mode: "text", limit: 8 });
      if (res && res.results.length > 0) {
        const target = res.results.find((r) => r.has_pair) ?? res.results[0];
        await selectAoi(target.aoi_key);
      }
    })();
  }, [run, selectAoi]);

  // Quietly seed the review queue from the rest of the featured results so the
  // analyst has a real backlog to work, without disturbing the open analysis.
  const seeded = useRef(false);
  useEffect(() => {
    if (seeded.current || !response) return;
    const keys = response.results
      .filter((r) => r.has_pair)
      .slice(1, 5)
      .map((r) => r.aoi_key);
    if (keys.length === 0) return;
    seeded.current = true;
    void (async () => {
      for (const key of keys) {
        try {
          await arc.verify(key);
          await refreshReview();
        } catch {
          /* a single AOI that cannot be verified should not stall seeding */
        }
      }
    })();
  }, [response, refreshReview]);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = queryText.trim();
    if (!text) return;
    void (async () => {
      const res = await run({ text, mode: "text", limit: 8 });
      if (res && res.results.length > 0) {
        const target = res.results.find((r) => r.has_pair) ?? res.results[0];
        await selectAoi(target.aoi_key);
      }
    })();
  };

  const onSelectResult = (r: SearchResult) => void selectAoi(r.aoi_key);

  const onResetQueue = useCallback(async () => {
    try {
      await arc.resetReview();
    } finally {
      await refreshReview();
    }
  }, [refreshReview]);

  // Keyboard: C confirms, R rejects the review item for the selected AOI.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (document.activeElement?.tagName ?? "").toLowerCase();
      if (tag === "input" || tag === "textarea" || e.metaKey || e.ctrlKey) return;
      if (!selectedKey) return;
      const item = reviewState?.items.find((i) => i.aoi_key === selectedKey);
      if (!item) return;
      if (e.key === "c" || e.key === "C") void decide(item.item_id, "confirm");
      else if (e.key === "r" || e.key === "R") void decide(item.item_id, "reject");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedKey, reviewState, decide]);

  const selectedResult = response?.results.find((r) => r.aoi_key === selectedKey) ?? null;
  const selectedPlace = verification?.place ?? selectedResult?.place ?? null;
  const facets = facetSummary(response);
  const pending = reviewState?.pending ?? 0;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-void text-ink">
      {/* ---------------------------------------------------------------- Header */}
      <header className="flex shrink-0 items-center gap-3 border-b border-edge bg-void/90 px-3 py-2.5 backdrop-blur sm:gap-4 sm:px-4">
        <Link to="/" aria-label="Back to the landing page" className="shrink-0">
          <Wordmark compact />
        </Link>
        <span className="hidden items-center gap-2 whitespace-nowrap text-[12px] md:flex">
          <span className="text-ink-dim">Studio</span>
          <span className="text-ink-faint/50">/</span>
          <span className="text-ink-faint">Search &amp; Review</span>
        </span>

        <form
          onSubmit={onSubmit}
          className="mx-auto flex min-w-0 max-w-3xl flex-1 items-center gap-2 rounded-xl border border-edge bg-hull/70 px-2 py-1.5 transition focus-within:border-signal/50"
        >
          <span className="shrink-0 rounded-md bg-signal/10 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-signal">
            Text
          </span>
          <input
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            placeholder={"Search the archive by meaning\u2026"}
            aria-label="Search the archive"
            className="min-w-0 flex-1 bg-transparent text-[13px] text-ink placeholder:text-ink-faint focus:outline-none"
          />
          {facets && (
            <span className="hidden max-w-[42%] shrink truncate text-[11px] text-ink-faint xl:inline">
              {facets}
            </span>
          )}
          <button
            type="submit"
            aria-label="Run search"
            className="grid h-7 w-7 shrink-0 place-items-center rounded-md text-ink-faint transition hover:bg-signal/10 hover:text-signal"
          >
            {searching ? <Spinner /> : <SearchIcon />}
          </button>
        </form>

        <div className="flex shrink-0 items-center gap-2 sm:gap-2.5">
          <OfflinePill />
          <button
            type="button"
            aria-label={`Review queue: ${pending} pending`}
            title={`${pending} pending review${pending === 1 ? "" : "s"}`}
            className="relative hidden h-8 w-8 place-items-center rounded-lg border border-edge text-ink-faint transition hover:border-edge-hi hover:text-ink sm:grid"
          >
            <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path
                d="M8 2a3.5 3.5 0 0 0-3.5 3.5c0 3-1.5 4-1.5 4h10s-1.5-1-1.5-4A3.5 3.5 0 0 0 8 2ZM6.5 13a1.5 1.5 0 0 0 3 0"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinejoin="round"
              />
            </svg>
            {pending > 0 && (
              <span className="tabular absolute -right-1 -top-1 grid h-4 min-w-4 place-items-center rounded-full bg-signal px-1 text-[9px] font-bold text-void">
                {pending}
              </span>
            )}
          </button>
          <PanelBoundary panel="The account menu">
            <AccountMenu onResetQueue={onResetQueue} />
          </PanelBoundary>
        </div>
      </header>

      {/* ------------------------------------------------------------------ Body */}
      <main className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-y-auto p-3 xl:grid-cols-[minmax(0,360px)_minmax(0,1fr)_minmax(0,380px)] xl:overflow-hidden">
        {/* Left: search results */}
        <div className="min-h-[70vh] [&>section]:h-full xl:min-h-0">
          <SearchResultsPanel
            response={response}
            busy={searching}
            selectedKey={selectedKey}
            verifyingKey={verifying ? verifyingKey : null}
            onSelect={onSelectResult}
          />
        </div>

        {/* Centre: map over verified-change analysis */}
        <div className="flex min-h-0 flex-col gap-3">
          <div className="h-[360px] shrink-0 xl:h-[46%] xl:min-h-[260px]">
            <ArcMap
              results={response?.results ?? []}
              selectedKey={selectedKey}
              onSelect={(key) => void selectAoi(key)}
            />
          </div>
          <div className="min-h-0 xl:flex-1 xl:overflow-y-auto">
            <ChangeAnalysisPanel
              verification={verification}
              busy={verifying}
              error={verifyError}
              place={selectedPlace}
            />
          </div>
        </div>

        {/* Right: review queue over provenance */}
        <div className="flex min-h-0 flex-col gap-3">
          <div className="min-h-[70vh] [&>section]:h-full xl:min-h-0 xl:flex-1">
            <ReviewQueuePanel
              state={reviewState}
              selectedKey={selectedKey}
              onSelect={(key) => void selectAoi(key)}
              onDecide={(itemId, decision) => void decide(itemId, decision)}
            />
          </div>
          <ProvenancePanel verification={verification} />
        </div>
      </main>
    </div>
  );
}
