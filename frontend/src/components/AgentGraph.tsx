import { motion } from "framer-motion";
import { formatDuration } from "../lib/format";
import { usePrefersReducedMotion } from "../lib/useReducedMotion";
import type { RunStatus, StageStatus, TraceStage } from "../lib/types";

const STATUS_STYLE: Record<
  StageStatus,
  { ring: string; dot: string; text: string; label: string }
> = {
  pending: {
    ring: "border-edge bg-hull/40",
    dot: "bg-edge-hi",
    text: "text-ink-faint",
    label: "waiting",
  },
  running: {
    ring: "border-signal/60 bg-signal/10",
    dot: "bg-signal",
    text: "text-signal",
    label: "running",
  },
  ok: {
    ring: "border-agree/50 bg-agree/10",
    dot: "bg-agree",
    text: "text-agree",
    label: "done",
  },
  partial: {
    ring: "border-uncertain/50 bg-uncertain/10",
    dot: "bg-uncertain",
    text: "text-uncertain",
    label: "partial",
  },
  skipped: {
    ring: "border-edge bg-hull/30",
    dot: "bg-ink-faint",
    text: "text-ink-faint",
    label: "skipped",
  },
  failed: {
    ring: "border-disagree/50 bg-disagree/10",
    dot: "bg-disagree",
    text: "text-disagree",
    label: "failed",
  },
  not_built: {
    ring: "border-dashed border-edge-hi bg-transparent",
    dot: "bg-transparent border border-edge-hi",
    text: "text-ink-faint",
    label: "not in this build",
  },
};

const RUN_LABEL: Record<RunStatus, string> = {
  running: "Running",
  completed: "Complete",
  refused: "Refused",
  failed: "Failed",
  cancelled: "Cancelled",
};

const RUN_TONE: Record<RunStatus, string> = {
  running: "text-signal",
  completed: "text-agree",
  refused: "text-uncertain",
  failed: "text-disagree",
  cancelled: "text-ink-faint",
};

/**
 * The pipeline, drawn from the stage list the server declared.
 *
 * Every stage is shown including the ones this build does not implement, drawn
 * with a dashed outline. A diagram that hides its gaps would misrepresent what
 * the system does, and the gaps close as the later stages land.
 */
export function AgentGraph({
  stages,
  status,
  durationMs,
}: {
  stages: TraceStage[];
  status: RunStatus;
  durationMs: number;
}) {
  const reduceMotion = usePrefersReducedMotion();
  const settled = stages.filter(
    (stage) => stage.status !== "pending" && stage.status !== "running",
  ).length;

  return (
    <section className="panel overflow-hidden" aria-label="Execution pipeline">
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 border-b border-edge px-4 py-2.5">
        <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
          Pipeline
        </h2>
        <div className="flex items-center gap-3">
          <span className="tabular text-[10px] text-ink-faint">
            {settled} / {stages.length} stages
          </span>
          <span
            className={`text-[10px] uppercase tracking-wider ${RUN_TONE[status]}`}
          >
            {RUN_LABEL[status]}
          </span>
          <span className="tabular text-[10px] text-ink-faint">
            {formatDuration(durationMs)}
          </span>
        </div>
      </div>

      {/*
        A wrapping grid, not a scrolling row.

        Seven fixed-width cards with connectors between them need about 1100px,
        which the centre column does not have once the sidebar is accounted for. So
        it grew a horizontal scrollbar and clipped the last two stages: the run
        appeared to end at "Resolve verdict" with no way to see that composing had
        happened, on the one panel whose job is to show the whole pipeline at a
        glance.

        The connectors went with the row. The ordinals carry the sequence, and a
        stage you can actually see beats a line pointing at one you cannot.
      */}
      <ol className="grid grid-cols-2 gap-2 px-4 py-3.5 sm:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-7">
        {stages.map((stage, index) => {
          const style = STATUS_STYLE[stage.status];

          return (
            <li
              key={stage.id}
              className="flex min-w-0"
              aria-label={`${stage.label}: ${style.label}`}
            >
              <div
                className={`relative w-full min-w-0 rounded-xl border px-2.5 py-2 transition-colors ${style.ring}`}
              >
                {stage.status === "running" && !reduceMotion && (
                  <motion.span
                    className="pointer-events-none absolute inset-0 rounded-xl border border-signal"
                    animate={{ opacity: [0.7, 0.15, 0.7] }}
                    transition={{ duration: 1.4, repeat: Infinity }}
                    aria-hidden="true"
                  />
                )}
                <div className="flex items-center gap-1.5">
                  <span
                    className={`h-1.5 w-1.5 shrink-0 rounded-full ${style.dot}`}
                    aria-hidden="true"
                  />
                  <span className="tabular text-[9px] text-ink-faint">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                </div>
                <p className="mt-1 break-words text-[11px] font-medium leading-tight text-ink">
                  {stage.label}
                </p>
                <p className={`mt-0.5 text-[9px] leading-tight ${style.text}`}>
                  {stage.status === "not_built"
                    ? style.label
                    : stage.duration_ms > 0
                      ? formatDuration(stage.duration_ms)
                      : style.label}
                </p>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
