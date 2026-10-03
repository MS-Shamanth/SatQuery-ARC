import type {
  ChangeVerification,
  ReviewItem,
  ReviewState,
  SearchResponse,
  SearchResult,
} from "../../lib/arc";

// ---------------------------------------------------------------------------
// Shared bits
// ---------------------------------------------------------------------------

function Spinner({ className = "" }: { className?: string }) {
  return (
    <span
      className={`inline-block animate-spin rounded-full border-2 border-signal/30 border-t-signal ${className}`}
    />
  );
}

const VERDICT_TONE: Record<string, { text: string; ring: string; chip: string }> = {
  supported: { text: "text-agree", ring: "border-agree/40", chip: "bg-agree/10 text-agree border-agree/30" },
  refuted: { text: "text-disagree", ring: "border-disagree/40", chip: "bg-disagree/10 text-disagree border-disagree/30" },
  inconclusive: { text: "text-uncertain", ring: "border-uncertain/40", chip: "bg-uncertain/10 text-uncertain border-uncertain/30" },
  unanswerable: { text: "text-ink-faint", ring: "border-edge", chip: "bg-panel text-ink-faint border-edge" },
};

function tagClass(kind: string): string {
  switch (kind) {
    case "change":
      return "border-agree/30 bg-agree/10 text-agree";
    case "water":
      return "border-signal/30 bg-signal/10 text-signal";
    case "quality":
      return "border-uncertain/30 bg-uncertain/10 text-uncertain";
    default:
      return "border-edge bg-hull/60 text-ink-dim";
  }
}

// ---------------------------------------------------------------------------
// Search results (left column)
// ---------------------------------------------------------------------------

export function SearchResultsPanel({
  response,
  busy,
  error = null,
  selectedKey,
  verifyingKey,
  onSelect,
}: {
  response: SearchResponse | null;
  busy: boolean;
  /** A failed search, shown in place of results so a backend fault is visible. */
  error?: string | null;
  selectedKey: string | null;
  verifyingKey: string | null;
  onSelect: (r: SearchResult) => void;
}) {
  return (
    <section className="panel flex min-h-0 flex-col" aria-label="Search results">
      <div className="flex items-center justify-between border-b border-edge px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold text-ink">Search Results</h2>
          <p className="tabular mt-0.5 text-[11px] text-ink-faint">
            {response ? `${response.total_indexed} sites indexed - top ${response.results.length}` : "-"}
          </p>
        </div>
        {busy && <Spinner className="h-4 w-4" />}
      </div>

      {response && (
        <p className="border-b border-edge px-4 py-2 text-[10px] leading-snug text-ink-faint">
          {response.interpreted_as}
        </p>
      )}

      <ul className="min-h-0 flex-1 divide-y divide-edge overflow-y-auto overscroll-contain">
        {(response?.results ?? []).map((r) => {
          const active = r.aoi_key === selectedKey;
          return (
            <li key={r.tile_id}>
              <button
                type="button"
                onClick={() => onSelect(r)}
                className={`flex w-full gap-3 px-3 py-3 text-left transition ${
                  active ? "bg-signal/5" : "hover:bg-panel-hi/40"
                }`}
              >
                <span className="tabular mt-0.5 w-4 shrink-0 text-[11px] text-ink-faint">
                  {r.rank}
                </span>
                <span className="relative h-14 w-14 shrink-0 overflow-hidden rounded-md border border-edge-hi">
                  <img src={r.thumbnail_url} alt="" className="h-full w-full object-cover" loading="lazy" />
                  {verifyingKey === r.aoi_key && (
                    <span className="absolute inset-0 grid place-items-center bg-void/60">
                      <Spinner className="h-4 w-4" />
                    </span>
                  )}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center justify-between gap-2">
                    <span className="truncate text-[13px] font-semibold text-ink">{r.place}</span>
                    <span className="tabular shrink-0 text-[13px] font-bold text-signal">
                      {r.match.toFixed(2)}
                    </span>
                  </span>
                  <span className="mt-0.5 block truncate text-[11px] text-ink-faint">
                    {r.acquisition_date ?? "n/a"} &middot; {r.sensor}
                    {r.gsd_m ? ` \u00b7 ${r.gsd_m} m` : ""}
                  </span>
                  <span className="mt-1.5 flex flex-wrap gap-1">
                    {r.tags.map((t, i) => (
                      <span
                        key={i}
                        className={`rounded border px-1.5 py-0.5 text-[9.5px] ${tagClass(t.kind)}`}
                      >
                        {t.label}
                      </span>
                    ))}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
        {response && response.results.length === 0 && (
          <li className="px-4 py-8 text-center text-[11px] text-ink-faint">
            No sites matched. Try a broader query or clear the filters.
          </li>
        )}
        {error && !busy && (
          <li role="alert" className="px-4 py-8 text-center text-[11px] leading-relaxed text-disagree">
            Search failed: {error}
          </li>
        )}
        {!response && !busy && !error && (
          <li className="px-4 py-8 text-center text-[11px] text-ink-faint">
            Search the archive to see results.
          </li>
        )}
      </ul>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Verified change analysis (centre, below the map)
// ---------------------------------------------------------------------------

function Frame({
  label,
  date,
  url,
  index,
  indexValue,
  highlight,
}: {
  label: string;
  date: string | null;
  url: string;
  index: string;
  indexValue: number | null;
  highlight?: boolean;
}) {
  return (
    <div className="flex-1">
      <div className={`relative overflow-hidden rounded-lg border ${highlight ? "border-agree/50" : "border-edge-hi"}`}>
        <img src={url} alt={label} className="aspect-[4/3] w-full object-cover" />
        <span className="absolute left-1.5 top-1.5 rounded bg-void/70 px-1.5 py-0.5 font-mono text-[9px] text-ink">
          {date ?? "n/a"} &middot; {label}
        </span>
      </div>
      <div className="mt-1 flex items-center justify-between text-[10px] text-ink-faint">
        <span>Sentinel-2</span>
        <span className={`tabular ${highlight ? "text-agree" : ""}`}>
          {index} {indexValue !== null ? indexValue.toFixed(2) : "-"}
        </span>
      </div>
    </div>
  );
}

function ConfidenceBars({ v }: { v: ChangeVerification }) {
  return (
    <div className="space-y-1.5">
      {v.confidence_components.map((c) => {
        const negative = c.contribution < 0;
        const filled = Math.min(1, Math.abs(c.contribution) / Math.max(c.weight, 0.01));
        return (
          <div key={c.name} className="grid grid-cols-[128px_1fr_34px] items-center gap-2">
            <span className="truncate text-[11px] text-ink-dim" title={c.label}>
              {c.label}
            </span>
            <span className="h-1.5 overflow-hidden rounded bg-white/8">
              <span
                className={`block h-full rounded ${negative ? "bg-disagree" : "bg-signal"}`}
                style={{ width: `${filled * 100}%` }}
              />
            </span>
            <span className={`tabular text-right text-[10px] ${negative ? "text-disagree" : "text-ink"}`}>
              {c.contribution >= 0 ? "+" : ""}
              {c.contribution.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function Firewall({ v }: { v: ChangeVerification }) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-[10px] uppercase tracking-wider text-ink-faint">
        <span>Confounder firewall</span>
        <span className="tabular">
          {v.confounders_cleared}/{v.confounders_total} cleared
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {v.confounders.map((c) => {
          const tone =
            c.verdict === "ruled_out"
              ? "border-agree/25 bg-agree/8 text-agree"
              : c.verdict === "likely"
                ? "border-disagree/30 bg-disagree/10 text-disagree"
                : c.verdict === "plausible"
                  ? "border-uncertain/30 bg-uncertain/10 text-uncertain"
                  : "border-edge bg-hull/60 text-ink-faint";
          const glyph = c.verdict === "ruled_out" ? "\u2713" : c.verdict === "not_tested" ? "\u2014" : "!";
          return (
            <span
              key={c.kind}
              className={`inline-flex items-center gap-1.5 rounded border px-2 py-1 text-[10.5px] ${tone}`}
              title={c.detail}
            >
              <span aria-hidden>{glyph}</span>
              {c.label}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function Timeline({ v }: { v: ChangeVerification }) {
  const obs = v.timeline;
  if (obs.length === 0) return null;
  const times = obs.map((o) => new Date(o.date).getTime());
  const min = Math.min(...times);
  const max = Math.max(...times);
  const span = Math.max(1, max - min);
  const pos = (d: string) => ((new Date(d).getTime() - min) / span) * 100;
  const years = Array.from(new Set(obs.map((o) => o.date.slice(0, 4))));

  return (
    <div className="border-t border-dashed border-edge pt-3">
      <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-wider text-ink-faint">
        <span>Usable observations &amp; earliest supported date</span>
        <span className="tabular">
          {obs.filter((o) => o.clear).length} clear / {obs.length} total
        </span>
      </div>
      <div className="relative h-9">
        <div className="absolute inset-x-0 top-4 h-0.5 bg-edge" />
        {obs.map((o) => (
          <div
            key={o.date}
            className="absolute top-[11px] -translate-x-1/2"
            style={{ left: `${pos(o.date)}%` }}
            title={`${o.date}${o.clear ? " (clear)" : " (cloudy)"}`}
          >
            <span
              className={`block h-2.5 w-2.5 rounded-full ${o.clear ? "bg-signal" : "bg-edge-hi"}`}
            />
          </div>
        ))}
        {obs
          .filter((o) => o.is_earliest_supported)
          .map((o) => (
            <div
              key={`flag-${o.date}`}
              className="absolute -top-1 -translate-x-1/2 text-center"
              style={{ left: `${pos(o.date)}%` }}
            >
              <span className="mx-auto block h-5 w-0.5 bg-agree" />
              <span className="whitespace-nowrap font-mono text-[9px] text-agree">{o.date}</span>
            </div>
          ))}
        <div className="absolute inset-x-0 top-[26px] flex justify-between font-mono text-[9px] text-ink-faint">
          <span>{years[0]}</span>
          {years.length > 1 && <span>{years[years.length - 1]}</span>}
        </div>
      </div>
      <div className="mt-1 flex gap-4 text-[9.5px] text-ink-faint">
        <span className="flex items-center gap-1"><i className="inline-block h-2 w-2 rounded-full bg-signal" /> clear</span>
        <span className="flex items-center gap-1"><i className="inline-block h-2 w-2 rounded-full bg-edge-hi" /> cloudy / masked</span>
        <span className="flex items-center gap-1"><i className="inline-block h-2 w-2 rounded-full bg-agree" /> earliest supported</span>
      </div>
    </div>
  );
}

export function ChangeAnalysisPanel({
  verification,
  busy,
  error,
  place,
}: {
  verification: ChangeVerification | null;
  busy: boolean;
  error: string | null;
  place: string | null;
}) {
  if (busy) {
    return (
      <section className="panel flex min-h-[280px] flex-col items-center justify-center gap-3 p-6 text-center">
        <Spinner className="h-6 w-6" />
        <p className="text-[12px] text-ink-dim">
          Verifying change{place ? ` at ${place}` : ""}
        </p>
        <p className="max-w-md text-[10px] leading-relaxed text-ink-faint">
          Readiness gate &rarr; change engine &rarr; confounder firewall &rarr; calibrated verdict.
          Fully offline; no data leaves the building.
        </p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel p-4">
        <p className="text-[11px] text-disagree">{error}</p>
      </section>
    );
  }

  if (!verification) {
    return (
      <section className="panel flex min-h-[220px] items-center justify-center p-6 text-center">
        <p className="max-w-sm text-[11px] leading-relaxed text-ink-faint">
          Select a search result to verify its change. SatQuery ARC compares the two
          acquisitions, runs the confounder firewall, and returns a calibrated verdict.
        </p>
      </section>
    );
  }

  const v = verification;
  const tone = VERDICT_TONE[v.verdict_label] ?? VERDICT_TONE.unanswerable;

  return (
    <section className={`panel overflow-hidden border ${tone.ring}`} aria-label="Verified change analysis">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge px-4 py-2.5">
        <h2 className="text-sm font-semibold text-ink">Verified Change Analysis</h2>
        <span className={`rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${tone.chip}`}>
          {v.headline}
        </span>
      </div>

      <div className="grid gap-4 p-4 lg:grid-cols-[300px_1fr]">
        <div>
          <div className="flex gap-2">
            <Frame label="before" date={v.before.acquisition_date} url={v.before.thumbnail_url}
              index={v.before.index_label} indexValue={v.before.index_value} />
            <Frame label="after" date={v.after.acquisition_date} url={v.after.thumbnail_url}
              index={v.after.index_label} indexValue={v.after.index_value} highlight />
          </div>
        </div>

        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="text-[10px] uppercase tracking-wider text-ink-faint">
                &Delta; {v.target_label} area
              </p>
              <p className={`tabular text-2xl font-bold leading-none ${v.direction === "increased" ? "text-agree" : v.direction === "decreased" ? "text-disagree" : "text-ink"}`}>
                {v.delta_display}
              </p>
              <p className="mt-1 text-[10px] text-ink-faint">
                {v.earliest_supported_date
                  ? <>earliest supported <span className="tabular text-ink-dim">{v.earliest_supported_date}</span> &middot; persists in {v.persists_in_scenes} clear scene{v.persists_in_scenes === 1 ? "" : "s"}</>
                  : "no supported earliest date on this pair"}
              </p>
            </div>
            <div className="text-right">
              <p className={`tabular text-3xl font-bold leading-none ${tone.text}`}>
                {(v.confidence * 100).toFixed(0)}%
              </p>
              <p className="text-[9px] uppercase tracking-wider text-ink-faint">confidence</p>
            </div>
          </div>

          <ConfidenceBars v={v} />
          <Firewall v={v} />
          <Timeline v={v} />
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Review queue + audit log (right column, top)
// ---------------------------------------------------------------------------

function StatusIcon({ status }: { status: ReviewItem["status"] }) {
  const map = {
    confirmed: { cls: "bg-agree/15 text-agree", glyph: "\u2713" },
    pending: { cls: "bg-signal/15 text-signal", glyph: "\u2022" },
    rejected: { cls: "bg-disagree/15 text-disagree", glyph: "\u2715" },
  } as const;
  const s = map[status];
  return (
    <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-md font-mono text-[11px] ${s.cls}`}>
      {s.glyph}
    </span>
  );
}

export function ReviewQueuePanel({
  state,
  selectedKey,
  onSelect,
  onDecide,
}: {
  state: ReviewState | null;
  selectedKey: string | null;
  onSelect: (aoiKey: string) => void;
  onDecide: (itemId: string, decision: "confirm" | "reject") => void;
}) {
  return (
    <section className="panel flex min-h-0 flex-col" aria-label="Review queue">
      <div className="flex items-center justify-between border-b border-edge px-4 py-3">
        <h2 className="text-sm font-semibold text-ink">Review Queue</h2>
        <span className="text-[10px] uppercase tracking-wider text-ink-faint">
          re-ranked by confidence
        </span>
      </div>

      <ul className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
        {(state?.items ?? []).map((it) => {
          const active = it.aoi_key === selectedKey;
          return (
            <li
              key={it.item_id}
              className={`rounded-lg border p-2.5 ${active ? "border-signal/50 bg-signal/5" : "border-edge bg-hull/40"}`}
            >
              <button type="button" onClick={() => onSelect(it.aoi_key)} className="flex w-full items-center gap-2.5 text-left">
                <StatusIcon status={it.status} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[12.5px] font-semibold text-ink">{it.place}</span>
                  <span className="block truncate text-[10.5px] text-ink-faint">
                    {it.change_hint} &middot; {it.delta_display}
                  </span>
                </span>
                <span className="tabular shrink-0 text-[11px] text-ink-dim">
                  {(it.confidence * 100).toFixed(0)}%
                </span>
              </button>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  onClick={() => onDecide(it.item_id, "confirm")}
                  className={`flex-1 rounded-md border px-2 py-1 text-[11px] font-semibold transition ${
                    it.status === "confirmed"
                      ? "border-agree bg-agree text-void"
                      : "border-agree/40 text-agree hover:bg-agree/10"
                  }`}
                >
                  Confirm <span className="font-mono opacity-60">C</span>
                </button>
                <button
                  type="button"
                  onClick={() => onDecide(it.item_id, "reject")}
                  className={`flex-1 rounded-md border px-2 py-1 text-[11px] font-semibold transition ${
                    it.status === "rejected"
                      ? "border-disagree bg-disagree/80 text-void"
                      : "border-disagree/40 text-disagree hover:bg-disagree/10"
                  }`}
                >
                  Reject <span className="font-mono opacity-60">R</span>
                </button>
              </div>
            </li>
          );
        })}
        {(!state || state.items.length === 0) && (
          <li className="px-2 py-6 text-center text-[11px] text-ink-faint">
            Verify a result to add it to the review queue.
          </li>
        )}
      </ul>

      {/* Audit log */}
      <div className="border-t border-edge px-3 py-2.5">
        <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-ink-faint">
          <span className="h-1.5 w-1.5 rounded-full bg-signal" /> Audit log
          {state && <span className="tabular ml-auto normal-case">{state.audit.length} entries</span>}
        </div>
        <ul className="max-h-28 space-y-1 overflow-y-auto font-mono text-[9.5px] leading-relaxed text-ink-faint">
          {(state?.audit ?? []).slice(0, 12).map((e) => (
            <li key={e.entry_id} className="truncate" title={`${e.action} - ${e.detail}`}>
              <span className="text-ink-dim">{e.at.slice(11, 19)}</span>{" "}
              <span className="text-signal/80">{e.action}</span> {e.target}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Provenance & export (right column, bottom)
// ---------------------------------------------------------------------------

function download(name: string, data: object) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export function ProvenancePanel({ verification }: { verification: ChangeVerification | null }) {
  const v = verification;
  const rows: [string, string | null][] = v
    ? [
        ["Before scene", v.before_scene_id],
        ["After scene", v.after_scene_id],
        ["Embedding", v.embedding_model],
        ["Change model", v.change_model],
        ["Index engine", v.index_engine],
        ["Checksum", v.checksum],
      ]
    : [];

  const geojson = () => {
    if (!v) return;
    const [w, s, e, n] = v.bounds_wgs84;
    download(`satquery-arc-${v.aoi_key}.geojson`, {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
          properties: {
            place: v.place,
            verdict: v.verdict_label,
            confidence: v.confidence,
            delta: v.delta_display,
            earliest_supported: v.earliest_supported_date,
            index: v.index_name,
            checksum: v.checksum,
          },
        },
      ],
    });
  };

  return (
    <section className="panel p-4" aria-label="Provenance and export">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-ink">Provenance &amp; Export</h2>
        <span className="text-[10px] uppercase tracking-wider text-ink-faint">traceable</span>
      </div>
      {v ? (
        <>
          <dl className="text-[11px]">
            {rows.map(([k, val]) => (
              <div key={k} className="flex justify-between gap-3 border-b border-edge/50 py-1 last:border-0">
                <dt className="text-ink-faint">{k}</dt>
                <dd className="tabular max-w-[62%] truncate text-right text-ink-dim" title={val ?? ""}>
                  {val ?? "-"}
                </dd>
              </div>
            ))}
          </dl>
          <div className="mt-3 flex flex-wrap gap-2">
            <button type="button" onClick={geojson} className="rounded-md border border-edge px-3 py-1.5 text-[11px] text-ink-dim transition hover:border-signal/50 hover:text-signal">
              &#8615; GeoJSON
            </button>
            <button type="button" onClick={() => v && download(`satquery-arc-${v.aoi_key}.json`, v)} className="rounded-md border border-edge px-3 py-1.5 text-[11px] text-ink-dim transition hover:border-signal/50 hover:text-signal">
              &#8615; Evidence JSON
            </button>
            <button type="button" onClick={() => window.print()} className="rounded-md border border-edge px-3 py-1.5 text-[11px] text-ink-dim transition hover:border-signal/50 hover:text-signal">
              &#8615; PDF (print)
            </button>
          </div>
        </>
      ) : (
        <p className="text-[11px] text-ink-faint">
          Verify a change to see its full provenance and export a georeferenced record.
        </p>
      )}
    </section>
  );
}
