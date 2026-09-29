import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { formatDuration } from "../lib/format";
import type { LogLine } from "../lib/useRun";
import type {
  Measurement,
  RunTrace,
  StageStatus,
  TraceStage,
  TraceStep,
} from "../lib/types";

const STEP_TONE: Record<StageStatus, string> = {
  pending: "border-edge",
  running: "border-signal",
  ok: "border-agree",
  partial: "border-uncertain",
  skipped: "border-edge-hi",
  failed: "border-disagree",
  not_built: "border-edge-hi",
};

const IMPLEMENTATION_LABEL: Record<string, string> = {
  deterministic: "computed",
  learned: "trained model",
  "llm-narration": "wording only",
};

/**
 * The audit trail, written as the run happens.
 *
 * Each measurement can be expanded to show the formula that produced it and
 * every value that formula consumed. That is the Never-Guess Rule made checkable
 * rather than asserted: any number on screen is two clicks from its derivation.
 */
export function RunTracePanel({
  trace,
  log,
  streaming,
  onCancel,
}: {
  trace: RunTrace;
  log: LogLine[];
  streaming: boolean;
  onCancel: () => void;
}) {
  /*
    Collapsed by default, and that is a change of mind worth stating.

    The trace is the product: every step, every parameter, every measurement with
    its formula. Printed open it runs to thousands of pixels, and the effect was
    the opposite of transparency, because the reader scrolled past all of it to
    reach the end of the page. Nothing is hidden that was not one click away
    before; the difference is that the reader now chooses what to read.
  */
  const [showAll, setShowAll] = useState(false);
  const [opened, setOpened] = useState<Set<string>>(new Set());
  const steps = trace.stages.reduce(
    (total, stage) => total + stage.steps.length,
    0,
  );

  return (
    <section className="panel overflow-hidden" aria-label="Execution trace">
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 border-b border-edge px-4 py-2.5">
        <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
          Execution trace
        </h2>
        <div className="flex items-center gap-3">
          <span className="tabular text-[10px] text-ink-faint">
            {trace.measurements.length} measurements &middot; {steps} steps
          </span>
          <button
            type="button"
            onClick={() => {
              setShowAll((value) => !value);
              // Clearing the per-stage set stops "Collapse all" leaving behind a
              // stage the reader opened by hand ten minutes ago.
              setOpened(new Set());
            }}
            aria-expanded={showAll}
            className={`rounded border px-2 py-0.5 text-[10px] transition ${
              showAll
                ? "border-signal/50 bg-signal/10 text-signal"
                : "border-edge text-ink-dim hover:border-edge-hi hover:text-ink"
            }`}
          >
            {showAll ? "Collapse all" : "Show all"}
          </button>
          {streaming && (
            <button
              type="button"
              onClick={onCancel}
              className="rounded border border-edge px-2 py-0.5 text-[10px] text-ink-dim transition hover:border-disagree/50 hover:text-disagree"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {trace.refusal_reasons.length > 0 && (
        <div className="border-b border-edge bg-uncertain/5 px-4 py-3">
          <p className="text-[10px] uppercase tracking-wider text-uncertain">
            The gate stopped this run
          </p>
          <ul className="mt-1 space-y-0.5">
            {trace.refusal_reasons.map((reason) => (
              <li key={reason} className="text-[11px] leading-snug text-ink-dim">
                {reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {trace.error && (
        <p
          role="alert"
          className="border-b border-edge px-4 py-3 text-[11px] leading-snug text-disagree"
        >
          {trace.error}
        </p>
      )}

      <div className="divide-y divide-edge">
        {trace.stages.map((stage) => (
          <StageBlock
            key={stage.id}
            stage={stage}
            trace={trace}
            // Everything, one stage the reader picked, or the stage in flight.
            // The last of those matters: collapsing the trace must not take away
            // the live view of what is happening right now.
            open={
              showAll || opened.has(stage.id) || stage.status === "running"
            }
            onToggle={() =>
              setOpened((current) => {
                const next = new Set(current);
                if (next.has(stage.id)) next.delete(stage.id);
                else next.add(stage.id);
                return next;
              })
            }
          />
        ))}
      </div>

      <LiveLog log={log} streaming={streaming} />
    </section>
  );
}

function StageBlock({
  stage,
  trace,
  open,
  onToggle,
}: {
  stage: TraceStage;
  trace: RunTrace;
  open: boolean;
  onToggle: () => void;
}) {
  const idle = stage.status === "pending";
  const count = stage.steps.length;

  return (
    <div className={`px-4 py-2.5 ${idle ? "opacity-50" : ""}`}>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full items-baseline justify-between gap-3 text-left"
      >
        <h3 className="min-w-0 text-[11px] font-medium text-ink">
          <span
            className={`mr-1.5 inline-block text-[9px] text-ink-faint transition-transform ${
              open ? "rotate-90" : ""
            }`}
            aria-hidden="true"
          >
            &#9656;
          </span>
          {stage.label}
        </h3>
        <span className="tabular flex shrink-0 items-baseline gap-2 text-[10px] text-ink-faint">
          {count > 0 && (
            <span>
              {count} step{count === 1 ? "" : "s"}
            </span>
          )}
          <span>{stage.duration_ms > 0 ? formatDuration(stage.duration_ms) : ""}</span>
        </span>
      </button>

      {open && (
        <p className="mt-0.5 text-[10px] leading-snug text-ink-faint">
          {stage.purpose}
        </p>
      )}

      {open && stage.unavailable_note && (
        <p className="mt-1.5 rounded border border-dashed border-edge-hi px-2 py-1.5 text-[10px] leading-snug text-ink-faint">
          {stage.unavailable_note}
        </p>
      )}

      {open && stage.steps.length > 0 && (
        <ul className="mt-2 space-y-1.5">
          <AnimatePresence initial={false}>
            {stage.steps.map((step) => (
              <motion.li
                key={step.id}
                initial={{ opacity: 0, x: -4 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.25 }}
                className={`border-l-2 pl-2.5 ${STEP_TONE[step.status]}`}
              >
                <StepRow step={step} trace={trace} />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}

      {open &&
        stage.notes.map((note) => (
          <p
            key={note}
            className="mt-1.5 text-[10px] leading-snug text-ink-faint"
          >
            {note}
          </p>
        ))}
    </div>
  );
}

function StepRow({ step, trace }: { step: TraceStep; trace: RunTrace }) {
  const [open, setOpen] = useState(false);
  const measurements = step.measurement_keys
    .map((key) => trace.measurements.find((item) => item.key === key))
    .filter((item): item is Measurement => Boolean(item));
  const expandable = measurements.length > 0 || step.notes.length > 0;

  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <p className="min-w-0 text-[11px] text-ink">
          <span className={step.tool ? "tabular" : ""}>{step.label}</span>
          {step.version && (
            <span className="ml-1 text-[10px] text-ink-faint">v{step.version}</span>
          )}
          {step.implementation && (
            <span className="ml-1.5 rounded bg-panel-hi px-1 text-[9px] text-ink-faint">
              {IMPLEMENTATION_LABEL[step.implementation] ?? step.implementation}
            </span>
          )}
        </p>
        <span className="tabular shrink-0 text-[10px] text-ink-faint">
          {step.duration_ms > 0 ? formatDuration(step.duration_ms) : ""}
        </span>
      </div>

      {step.detail && (
        <p className="text-[10px] leading-snug text-ink-dim">{step.detail}</p>
      )}

      {step.reason && (
        <p className="text-[10px] leading-snug text-uncertain">{step.reason}</p>
      )}

      {Object.keys(step.parameters).length > 0 && (
        <p className="tabular mt-0.5 text-[9px] text-ink-faint">
          {Object.entries(step.parameters)
            .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
            .join("  ")}
        </p>
      )}

      {expandable && (
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
          className="mt-1 text-[9px] uppercase tracking-wider text-ink-faint transition hover:text-signal"
        >
          {open ? "Hide" : "Show"}
          {measurements.length > 0 ? ` ${measurements.length} measurements` : " notes"}
        </button>
      )}

      {open && (
        <div className="mt-1.5 space-y-1.5">
          {step.notes.map((note) => (
            <p key={note} className="text-[10px] leading-snug text-ink-faint">
              {note}
            </p>
          ))}
          {measurements.map((item) => (
            <MeasurementRow key={item.key} measurement={item} />
          ))}
        </div>
      )}
    </div>
  );
}

/** A number, with the formula and inputs that produced it one click away. */
function MeasurementRow({ measurement }: { measurement: Measurement }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded border border-edge bg-hull/40 px-2 py-1.5">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-baseline justify-between gap-2 text-left"
      >
        <span className="min-w-0 text-[10px] text-ink-dim">{measurement.label}</span>
        <span className="tabular shrink-0 text-[11px] text-signal">
          {measurement.value.toFixed(measurement.precision)}
          <span className="ml-0.5 text-[9px] text-ink-faint">
            {measurement.unit}
          </span>
        </span>
      </button>
      {open && (
        <dl className="mt-1.5 space-y-1 border-t border-edge pt-1.5">
          <div>
            <dt className="text-[9px] uppercase tracking-wider text-ink-faint">
              Formula
            </dt>
            <dd className="tabular text-[10px] leading-snug text-ink-dim">
              {measurement.formula}
            </dd>
          </div>
          {Object.keys(measurement.inputs).length > 0 && (
            <div>
              <dt className="text-[9px] uppercase tracking-wider text-ink-faint">
                Inputs
              </dt>
              <dd className="tabular text-[10px] leading-snug text-ink-dim">
                {Object.entries(measurement.inputs)
                  .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
                  .join("  ")}
              </dd>
            </div>
          )}
          <div>
            <dt className="text-[9px] uppercase tracking-wider text-ink-faint">
              Source
            </dt>
            <dd className="tabular text-[10px] text-ink-dim">
              {measurement.source_tool} v{measurement.source_version}
              {measurement.method ? ` \u00b7 ${measurement.method}` : ""}
            </dd>
          </div>
        </dl>
      )}
    </div>
  );
}

function LiveLog({ log, streaming }: { log: LogLine[]; streaming: boolean }) {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [log.length]);

  if (log.length === 0) return null;

  return (
    <div className="border-t border-edge bg-abyss/60">
      <div className="flex items-center gap-2 px-4 py-2">
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            streaming ? "bg-signal" : "bg-edge-hi"
          }`}
          aria-hidden="true"
        />
        <p className="text-[10px] uppercase tracking-wider text-ink-faint">
          {streaming ? "Live" : "Run log"}
        </p>
      </div>
      <div
        className="max-h-44 overflow-y-auto px-4 pb-3"
        role="log"
        aria-live="polite"
        aria-label="Run log"
      >
        {log.map((line) => (
          <p
            key={line.seq}
            className="tabular text-[10px] leading-relaxed text-ink-faint"
          >
            <span className="text-edge-hi">
              {new Date(line.at).toLocaleTimeString("en-GB", { hour12: false })}
            </span>{" "}
            {line.text}
          </p>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}
