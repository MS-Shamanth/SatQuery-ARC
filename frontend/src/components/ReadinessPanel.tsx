import { AnimatePresence, motion } from "framer-motion";
import { useState, type CSSProperties } from "react";
import { formatDuration } from "../lib/format";
import type { CheckStatus, ReadinessReport, ReadinessVerdict } from "../lib/types";

const STATUS_DOT: Record<CheckStatus, string> = {
  pass: "bg-agree",
  warn: "bg-uncertain",
  fail: "bg-disagree",
  not_applicable: "bg-ink-faint/50",
};

const VERDICT: Record<
  ReadinessVerdict,
  { title: string; blurb: string; border: string; text: string; bg: string }
> = {
  ready: {
    title: "Ready",
    blurb: "Every check cleared. Analysis can proceed on these inputs.",
    border: "border-agree/40",
    text: "text-agree",
    bg: "bg-agree/5",
  },
  ready_with_warnings: {
    title: "Ready, with caveats",
    blurb:
      "Analysis can proceed. The flagged conditions are carried into the confounder tests rather than ignored.",
    border: "border-uncertain/40",
    text: "text-uncertain",
    bg: "bg-uncertain/5",
  },
  refused: {
    title: "Refused",
    blurb:
      "These inputs cannot support a trustworthy answer. Producing one anyway would mean reporting a number that does not mean what it appears to.",
    border: "border-disagree/40",
    text: "text-disagree",
    bg: "bg-disagree/5",
  },
};

function StatusIcon({ status }: { status: CheckStatus }) {
  if (status === "pass") {
    return (
      <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
        <path
          d="M1.5 5.5 4 8l4.5-6"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
    );
  }
  if (status === "warn") {
    return (
      <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
        <path d="M5 1 9.5 9h-9z" fill="none" stroke="currentColor" strokeWidth="1.2" />
        <path d="M5 4v2.2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      </svg>
    );
  }
  if (status === "fail") {
    return (
      <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
        <path
          d="M2 2l6 6M8 2l-6 6"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
    );
  }
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
      <path d="M2 5h6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function CheckRow({
  check,
  index,
  className = "",
}: {
  check: ReadinessReport["checks"][number];
  index: number;
  /** Layout-only additions, such as the borders of the two-column list. */
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const detailId = `check-detail-${check.id.replace(/\./g, "-")}`;

  return (
    <motion.li
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05, duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className={`border-b border-edge/60 last:border-0 ${className}`}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls={detailId}
        className="flex w-full items-center gap-2.5 px-3 py-2 text-left transition hover:bg-panel-hi/40"
      >
        <span
          className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full ${
            STATUS_DOT[check.status]
          } ${check.status === "not_applicable" ? "text-void/70" : "text-void"}`}
        >
          <StatusIcon status={check.status} />
        </span>
        <span className="min-w-0 flex-1 truncate text-[11px] text-ink">
          {check.label}
        </span>
        <span
          className={`tabular shrink-0 text-[11px] ${
            check.status === "not_applicable" ? "text-ink-faint" : "text-signal"
          }`}
        >
          {check.measured}
        </span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={detailId}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="space-y-1.5 px-3 pb-2.5 pl-9">
              <p className="text-[11px] leading-snug text-ink-dim">{check.message}</p>
              {check.threshold && (
                <p className="text-[10px] text-ink-faint">
                  Threshold:{" "}
                  <span className="tabular text-ink-dim">{check.threshold}</span>
                </p>
              )}
              {check.method && (
                <p className="text-[10px] text-ink-faint">
                  Measured by: <span className="text-ink-dim">{check.method}</span>
                </p>
              )}
              {check.applies_to.length > 0 && (
                <p className="text-[10px] text-ink-faint">
                  Applies to: {check.applies_to.join(", ")}
                </p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.li>
  );
}

function RadarSweep() {
  return (
    <div className="flex flex-col items-center gap-2.5 py-8" role="status">
      <div className="relative h-12 w-12 rounded-full border border-edge">
        <div className="absolute inset-0 animate-[var(--animate-sweep)]">
          <div
            className="absolute left-1/2 top-1/2 h-1/2 w-px origin-top"
            style={{
              background: "linear-gradient(to bottom, var(--color-signal), transparent)",
            }}
          />
        </div>
        <div className="absolute inset-[30%] rounded-full border border-edge/60" />
      </div>
      <p className="text-[11px] text-ink-dim">Checking data readiness</p>
      <p className="text-[10px] text-ink-faint">
        Reading projections, bands, cloud cover, and alignment
      </p>
    </div>
  );
}

interface Props {
  report: ReadinessReport | null;
  loading: boolean;
  error: string | null;
}

export function ReadinessPanel({ report, loading, error }: Props) {
  const [showNa, setShowNa] = useState(false);

  if (loading && !report) {
    return (
      <section className="panel" aria-label="Data readiness">
        <RadarSweep />
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel p-4" aria-label="Data readiness">
        <p className="text-[11px] text-disagree">{error}</p>
      </section>
    );
  }

  if (!report) return null;

  const verdict = VERDICT[report.verdict];
  const applicable = report.checks.filter((c) => c.status !== "not_applicable");
  const notApplicable = report.checks.filter((c) => c.status === "not_applicable");
  const counts = {
    pass: applicable.filter((c) => c.status === "pass").length,
    warn: applicable.filter((c) => c.status === "warn").length,
    fail: applicable.filter((c) => c.status === "fail").length,
  };
  // Two columns once the panel is wide enough that a single row would put a
  // check's name and its reading a monitor's width apart. Filled top to bottom,
  // so the list still reads in the order the checks were run.
  const rows = Math.ceil(applicable.length / 2);

  return (
    <section
      className="panel panel-raised @container overflow-hidden"
      aria-label="Data readiness"
    >
      <div className="flex items-center justify-between border-b border-edge px-4 py-2.5">
        <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
          Data readiness gate
        </h2>
        <span className="tabular text-[10px] text-ink-faint">
          {formatDuration(report.computed_ms)}
        </span>
      </div>

      <motion.div
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        className={`border-b ${verdict.border} ${verdict.bg} px-4 py-3`}
      >
        <div className="flex items-center gap-2.5">
          <p className={`text-sm font-semibold ${verdict.text}`}>{verdict.title}</p>
          <span className="tabular flex items-center gap-2 text-[10px] text-ink-faint">
            <span className="text-agree">{counts.pass} pass</span>
            {counts.warn > 0 && (
              <span className="text-uncertain">{counts.warn} warn</span>
            )}
            {counts.fail > 0 && (
              <span className="text-disagree">{counts.fail} fail</span>
            )}
          </span>
        </div>
        <p className="mt-1 text-[11px] leading-snug text-ink-dim">{verdict.blurb}</p>
      </motion.div>

      {report.refusal_reasons.length > 0 && (
        <div className="border-b border-edge px-4 py-3">
          <p className="mb-1.5 text-[10px] uppercase tracking-wider text-disagree">
            Why this was refused
          </p>
          <ul className="space-y-1">
            {report.refusal_reasons.map((reason) => (
              <li key={reason} className="text-[11px] leading-snug text-ink-dim">
                &middot; {reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {report.requirements.length > 0 && (
        <div className="border-b border-edge bg-hull/40 px-4 py-3">
          <p className="mb-1.5 text-[10px] uppercase tracking-wider text-signal">
            What would be needed
          </p>
          <ul className="space-y-2">
            {report.requirements.map((requirement) => (
              <li key={requirement.what}>
                <p className="text-[11px] font-medium text-ink">{requirement.what}</p>
                <p className="text-[10px] leading-snug text-ink-faint">
                  {requirement.why}
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}

      <ul
        className="@4xl:grid @4xl:grid-flow-col @4xl:grid-cols-2 @4xl:grid-rows-[repeat(var(--rows),auto)]"
        style={{ "--rows": String(rows) } as CSSProperties}
      >
        {applicable.map((check, index) => (
          <CheckRow
            key={check.id}
            check={check}
            index={index}
            className={
              index >= rows
                ? "@4xl:border-l"
                : index === rows - 1
                  ? "@4xl:border-b-0"
                  : ""
            }
          />
        ))}
      </ul>

      {notApplicable.length > 0 && (
        <div className="border-t border-edge">
          <button
            type="button"
            onClick={() => setShowNa((v) => !v)}
            aria-expanded={showNa}
            className="w-full px-3 py-2 text-left text-[10px] uppercase tracking-wider text-ink-faint transition hover:text-ink-dim"
          >
            {showNa ? "Hide" : "Show"} {notApplicable.length} check
            {notApplicable.length === 1 ? "" : "s"} that do not apply
          </button>
          {showNa && (
            <ul>
              {notApplicable.map((check, index) => (
                <CheckRow key={check.id} check={check} index={index} />
              ))}
            </ul>
          )}
        </div>
      )}

      {report.seasonal_risk && (
        <div className="border-t border-uncertain/30 bg-uncertain/5 px-4 py-2.5">
          <p className="text-[10px] uppercase tracking-wider text-uncertain">
            Carried forward
          </p>
          <p className="mt-1 text-[11px] leading-snug text-ink-dim">
            Seasonal offset of{" "}
            <span className="tabular text-uncertain">
              {report.month_of_year_delta ?? "?"} month
              {report.month_of_year_delta === 1 ? "" : "s"}
            </span>{" "}
            recorded. The seasonality confounder test will run against any
            vegetation change found here.
          </p>
        </div>
      )}
    </section>
  );
}
