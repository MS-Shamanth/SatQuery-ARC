import { AnimatePresence, motion } from "framer-motion";
import { Fragment, useState } from "react";
import type { ComponentProbe, HealthResponse, ProbeStatus } from "../lib/types";

const STATUS_COLOR: Record<ProbeStatus, string> = {
  ok: "bg-agree",
  skipped: "bg-ink-faint",
  unknown: "bg-ink-faint",
  // Amber, not red. A spent free allowance is a documented fallback taking over,
  // not a fault, and it needs no action beyond waiting for the reset.
  limited: "bg-uncertain",
  missing: "bg-uncertain",
  invalid: "bg-disagree",
  error: "bg-disagree",
};

const STATUS_WORD: Partial<Record<ProbeStatus, string>> = {
  limited: "rate limited",
  missing: "not configured",
  skipped: "standby",
};

const OVERALL_LABEL: Record<HealthResponse["status"], string> = {
  ok: "All systems nominal",
  degraded: "Operating with fallbacks",
  down: "Geospatial engine unavailable",
};

const COMPONENT_LABEL: Record<string, string> = {
  raster_stack: "Geospatial compute",
  // This had no entry, so the component key was printed raw and ran straight into
  // its status: "llm_providererror". Meanwhile the Gemini row was the one labelled
  // "Language model", which made the standby provider look like the live one.
  llm_provider: "Language model",
  gemini: "Gemini (standby)",
  copernicus: "Copernicus (SAR)",
  earth_search: "Earth Search (optical)",
};

const CAPABILITY_LABEL: Record<string, string> = {
  "copernicus-sentinel-1-grd": "Sentinel-1 GRD (real)",
  "aws-requester-pays-sentinel-1-grd": "Sentinel-1 GRD via AWS (real)",
  "simulated-sar": "Simulated sigma-0 (labelled)",
  "earth-search-sentinel-2-l2a": "Sentinel-2 L2A (real)",
  synthetic: "Synthetic scenes",
  gemini: "Gemini structured output",
  "language-model": "Planned by the language model",
  "offline-rule-router": "Offline rule router",
  cache: "Reused from cache",
};

interface Props {
  health: HealthResponse | null;
  error: string | null;
  loading: boolean;
  onRefresh: (deep?: boolean) => void;
}

/**
 * The paths that are actually carrying work, named by what they are.
 *
 * Read from ``capabilities`` rather than from the probe details, because that is
 * the backend's own answer to "what will the next run use" and it is already
 * resolved: which model replied, which SAR source, whether contracts come from a
 * model or the rule router. The probe strings are diagnostics about how it found
 * out, which is a different question.
 */
function activePaths(health: HealthResponse): [string, string][] {
  const caps = health.capabilities;
  const rows: [string, string][] = [];

  const models = caps.llm_models ?? [];
  if (caps.llm_provider && caps.llm_provider !== "none") {
    // The model that answered, not the one that was asked for, and stripped of
    // the provider prefix the chain adds since the provider is already named.
    const model = models[0]?.split("/").slice(-1)[0];
    rows.push([
      "Language model",
      model ? `${caps.llm_provider} \u00b7 ${model}` : caps.llm_provider,
    ]);
  } else {
    rows.push(["Language model", "none, using the offline rule router"]);
  }

  rows.push([
    "Optical",
    CAPABILITY_LABEL[caps.optical_source] ?? caps.optical_source,
  ]);
  rows.push(["SAR", CAPABILITY_LABEL[caps.sar_source] ?? caps.sar_source]);
  rows.push([
    "Contracts",
    CAPABILITY_LABEL[caps.contract_source] ?? caps.contract_source,
  ]);
  rows.push([
    "Measurement",
    caps.geospatial_compute ? "rasterio / GDAL" : "unavailable",
  ]);

  return rows;
}

export function HealthPill({ health, error, loading, onRefresh }: Props) {
  const [open, setOpen] = useState(false);
  const [detailed, setDetailed] = useState(false);

  const dotClass = error
    ? "bg-disagree"
    : loading || !health
      ? "bg-ink-faint"
      : health.status === "ok"
        ? "bg-agree"
        : health.status === "degraded"
          ? "bg-uncertain"
          : "bg-disagree";

  const label = error
    ? "Backend unreachable"
    : loading || !health
      ? "Checking systems"
      : OVERALL_LABEL[health.status];

  const components: ComponentProbe[] = health ? Object.values(health.components) : [];
  const inUse = health ? activePaths(health) : [];
  // Only genuine problems. "skipped" is a provider deliberately not checked, and
  // "limited" is a spent free allowance with a fallback already carrying the work:
  // neither is something anyone needs to act on.
  const attention = components.filter((probe) =>
    ["invalid", "error", "missing"].includes(probe.status),
  );

  return (
    // Positioned from sm up only. On a phone the details panel anchors to the
    // enclosing header instead, so it can span the screen rather than hang off
    // its left edge from a pill that sits near the right one.
    <div className="sm:relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`System status: ${label}. Toggle details.`}
        className="panel flex items-center gap-2.5 px-3 py-2 text-xs text-ink-dim transition hover:border-edge-hi hover:text-ink sm:px-3.5"
      >
        <span className="relative flex h-2 w-2 shrink-0">
          <span className={`absolute inset-0 rounded-full ${dotClass}`} />
          {!error && health?.status === "ok" && (
            <span className="absolute inset-0 animate-[var(--animate-pulse-ring)] rounded-full bg-agree/70" />
          )}
        </span>
        {/* The dot carries the state on a phone; the button's label still names it. */}
        <span className="hidden whitespace-nowrap sm:inline">{label}</span>
        <svg
          width="10"
          height="10"
          viewBox="0 0 10 10"
          className={`transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden="true"
        >
          <path
            d="M1 3.5 5 7.5 9 3.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.4"
          />
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ duration: 0.18 }}
            className="panel panel-raised absolute inset-x-3 top-full z-50 mt-2 max-h-[calc(100dvh-5rem)] overflow-y-auto p-4 text-left sm:inset-x-auto sm:right-0 sm:top-auto sm:w-[26rem]"
          >
            {error ? (
              <p className="text-xs leading-relaxed text-disagree">
                {error}
                <br />
                <span className="text-ink-faint">
                  Start it with{" "}
                  <code className="tabular text-ink-dim">
                    uvicorn app.main:app --reload
                  </code>{" "}
                  from <code className="tabular text-ink-dim">satquery/backend</code>.
                </span>
              </p>
            ) : (
              <>
                {/*
                  What is in use, and nothing else.

                  This listed every probe with its full detail string, which meant
                  the panel led with a rasterio build number and a paragraph of
                  OpenRouter quota prose about a provider that is not even being
                  used. None of that answers the question someone opens a status
                  light to ask, which is "what is running right now".

                  So: the working paths, named by what they are. Anything actually
                  wrong gets its own short section below, because hiding a failure
                  to keep a panel tidy would be the worse mistake. The raw probe text
                  is still here, one click away, for when it is the thing you want.
                */}
                <p className="mb-2 text-[10px] uppercase tracking-wider text-ink-faint">
                  In use
                </p>
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-[11px]">
                  {inUse.map(([label, value]) => (
                    <Fragment key={label}>
                      <dt className="whitespace-nowrap text-ink-faint">{label}</dt>
                      <dd className="min-w-0 text-ink-dim">{value}</dd>
                    </Fragment>
                  ))}
                </dl>

                {attention.length > 0 && (
                  <>
                    <div className="hairline my-3" />
                    <p className="mb-2 text-[10px] uppercase tracking-wider text-uncertain">
                      Needs attention
                    </p>
                    <ul className="space-y-1.5">
                      {attention.map((probe) => (
                        <li key={probe.component} className="flex gap-2.5">
                          <span
                            className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${
                              STATUS_COLOR[probe.status]
                            }`}
                          />
                          <p className="min-w-0 text-[11px] text-ink-dim">
                            {COMPONENT_LABEL[probe.component] ?? probe.component}
                            <span className="ml-1.5 text-[10px] uppercase text-ink-faint">
                              {STATUS_WORD[probe.status] ?? probe.status}
                            </span>
                          </p>
                        </li>
                      ))}
                    </ul>
                  </>
                )}

                <button
                  type="button"
                  onClick={() => setDetailed((value) => !value)}
                  aria-expanded={detailed}
                  className="mt-3 text-[10px] text-ink-faint transition hover:text-ink"
                >
                  {detailed ? "Hide probe detail" : "Probe detail"}
                </button>

                {detailed && (
                  <ul className="mt-2 space-y-2 border-t border-edge pt-2">
                    {components.map((probe) => (
                      <li key={probe.component} className="flex gap-2.5">
                        <span
                          className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${
                            STATUS_COLOR[probe.status]
                          }`}
                        />
                        <div className="min-w-0">
                          <p className="text-[11px] font-medium text-ink">
                            {COMPONENT_LABEL[probe.component] ?? probe.component}
                            <span className="tabular ml-2 text-[10px] uppercase text-ink-faint">
                              {STATUS_WORD[probe.status] ?? probe.status}
                            </span>
                          </p>
                          <p className="text-[10px] leading-snug text-ink-faint">
                            {probe.detail}
                          </p>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            )}

            <button
              type="button"
              onClick={() => onRefresh(true)}
              className="mt-3.5 w-full rounded-lg border border-edge px-3 py-1.5 text-[11px] text-ink-dim transition hover:border-signal/50 hover:text-signal"
            >
              Re-probe dependencies
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
