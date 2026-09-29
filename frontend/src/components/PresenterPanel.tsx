import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { DEMOS, type DemoBeat, type DemoScript } from "../lib/demos";
import { usePrefersReducedMotion } from "../lib/useReducedMotion";

export type PresenterPhase =
  | "idle"
  | "loading"
  | "drafting"
  | "awaiting_approval"
  | "running"
  | "answered";

interface Props {
  /** Which scripted demo is armed, if any. */
  active: DemoScript | null;
  phase: PresenterPhase;
  /** Scene keys the sample library actually has on disk. */
  availableKeys: Set<string>;
  onStart: (demo: DemoScript) => void | Promise<void>;
  onClear: () => void;
  busy?: boolean;
}

const PHASE_LABEL: Record<PresenterPhase, string> = {
  idle: "not started",
  loading: "loading the imagery",
  drafting: "drafting the plan",
  awaiting_approval: "waiting for you to approve the plan",
  running: "measuring",
  answered: "answered",
};

// Which part of the script is live during each phase, so the beats on screen are
// the ones the presenter needs now rather than the whole script at once.
const PHASE_BEATS: Record<PresenterPhase, DemoBeat["at"][]> = {
  idle: ["imagery"],
  loading: ["imagery", "gate"],
  drafting: ["gate", "contract"],
  awaiting_approval: ["contract"],
  running: ["running"],
  answered: ["answer", "packet"],
};

const BEAT_LABEL: Record<DemoBeat["at"], string> = {
  imagery: "In the imagery panel",
  gate: "In the readiness gate",
  contract: "In the analysis contract",
  running: "While the specialists run",
  answer: "In the answer",
  packet: "In the evidence packet",
};

/**
 * Presenter mode.
 *
 * It removes the fumbling, not the guarantees: loading the scene and drafting the
 * plan are automated, and then it stops. Approving the contract stays a click,
 * because "nothing touches a pixel until you accept the plan" is the whole point
 * of having a contract, and a demo that skipped past it would be disproving the
 * claim it was meant to demonstrate.
 *
 * The beats never quote a number. A script that promised a figure would be an
 * unmeasured number on screen, and it would go stale the moment the imagery did.
 */
export function PresenterPanel({
  active,
  phase,
  availableKeys,
  onStart,
  onClear,
  busy,
}: Props) {
  const reduceMotion = usePrefersReducedMotion();
  const [open, setOpen] = useState(false);

  const live = active
    ? active.beats.filter((beat) => PHASE_BEATS[phase].includes(beat.at))
    : [];

  return (
    <section className="panel overflow-hidden" aria-label="Presenter mode">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-4 py-2.5 text-left transition hover:bg-panel-hi/40"
      >
        <span className="flex items-baseline gap-2">
          <span className="text-[10px] uppercase tracking-wider text-ink-faint">
            Presenter
          </span>
          {active && (
            <span className="text-[11px] text-signal">
              {active.ordinal}. {active.title}
            </span>
          )}
        </span>
        <span className="tabular text-[10px] text-ink-faint">
          {active ? PHASE_LABEL[phase] : `${DEMOS.length} demos`}
        </span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: reduceMotion ? 0 : 0.28 }}
            className="border-t border-edge"
          >
            <div className="px-4 py-3">
              <p className="text-[10px] leading-snug text-ink-faint">
                Loads the scene and drafts the plan. Approving it stays your
                click, because nothing measures anything until you accept.
              </p>
            </div>

            <ul className="divide-y divide-edge border-t border-edge">
              {DEMOS.map((demo) => {
                const cached = availableKeys.has(demo.sampleKey);
                const current = active?.id === demo.id;
                return (
                  <li
                    key={demo.id}
                    className={`px-4 py-3 ${current ? "bg-signal/5" : ""}`}
                  >
                    <div className="flex items-baseline gap-2">
                      <span className="tabular text-[11px] text-ink-faint">
                        {demo.ordinal}
                      </span>
                      <p className="text-[11px] font-semibold text-ink">
                        {demo.title}
                      </p>
                    </div>
                    <p className="mt-1 text-[10px] leading-snug text-ink-dim">
                      {demo.premise}
                    </p>
                    <p className="mt-1.5 text-[10px] leading-snug text-ink-faint">
                      <span className="text-ink-dim">Expect:</span> {demo.expect}
                    </p>
                    {demo.followUp && (
                      <p className="mt-1 text-[10px] leading-snug text-signal/80">
                        Then the run offers a second scene that settles it.
                      </p>
                    )}
                    <div className="mt-2 flex items-center gap-2">
                      <button
                        type="button"
                        disabled={!cached || busy}
                        onClick={() => void onStart(demo)}
                        className="rounded-lg bg-signal px-3 py-1.5 text-[11px] font-semibold text-void transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        {current && phase !== "idle" ? "Restart" : "Set up"}
                      </button>
                      {!cached && (
                        <span className="text-[10px] text-uncertain">
                          not cached &mdash; run fetch_samples.py
                        </span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>

            {active && (
              <div className="border-t border-edge bg-hull/40 px-4 py-3">
                <div className="flex items-baseline justify-between gap-2">
                  <p className="text-[10px] uppercase tracking-wider text-signal">
                    What to point at now
                  </p>
                  <button
                    type="button"
                    onClick={onClear}
                    className="text-[10px] text-ink-faint transition hover:text-signal"
                  >
                    leave presenter
                  </button>
                </div>
                {live.length === 0 ? (
                  <p className="mt-1.5 text-[10px] text-ink-faint">
                    Nothing scripted for this step.
                  </p>
                ) : (
                  <ul className="mt-1.5 space-y-2">
                    {live.map((beat) => (
                      <li key={`${beat.at}-${beat.look}`}>
                        <p className="text-[9px] uppercase tracking-wider text-ink-faint">
                          {BEAT_LABEL[beat.at]}
                        </p>
                        <p className="text-[11px] leading-snug text-ink">
                          {beat.look}
                        </p>
                        <p className="mt-0.5 text-[10px] leading-snug text-ink-dim">
                          {beat.why}
                        </p>
                      </li>
                    ))}
                  </ul>
                )}

                {active.followUp && phase === "answered" && (
                  <div className="mt-3 rounded-lg border border-signal/30 bg-signal/5 px-3 py-2">
                    <p className="text-[9px] uppercase tracking-wider text-signal">
                      The second run
                    </p>
                    <p className="mt-0.5 text-[10px] leading-snug text-ink-dim">
                      {active.followUp.why}
                    </p>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
