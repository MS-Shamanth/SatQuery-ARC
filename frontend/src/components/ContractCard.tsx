import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { formatDuration } from "../lib/format";
import { usePrefersReducedMotion } from "../lib/useReducedMotion";
import type {
  AnalysisContract,
  ContractSource,
  ContractTaskType,
} from "../lib/types";

const TASK_LABEL: Record<ContractTaskType, string> = {
  scene_description: "Scene description",
  visual_question: "Visual question",
  region_grounding: "Region grounding",
  change_detection: "Change detection",
  change_question: "Change question",
  cross_modal_extraction: "Cross-modal extraction",
  claim_investigation: "Claim investigation",
};

const SOURCE_LABEL: Record<ContractSource, string> = {
  "language-model": "drafted by the language model",
  gemini: "drafted by the language model",
  "offline-rule-router": "drafted by the offline rule router",
  cache: "reused from cache",
};

// After this long the wait is worth explaining rather than just animating.
const PATIENCE_SECONDS = 4;

/**
 * The wait before a contract exists.
 *
 * It says what is being waited on, because the wait can be long and honest: the
 * provider is a pool of free models and a queued one has taken twenty-five
 * seconds before the offline planner took over. An indeterminate bar on its own
 * reads as "stuck" at that length, and a user who force-reloads never sees that
 * the fallback would have worked.
 */
function DraftingCard() {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const ticker = window.setInterval(
      () => setSeconds((elapsed) => elapsed + 1),
      1000,
    );
    return () => window.clearInterval(ticker);
  }, []);

  return (
    <section className="panel p-4" aria-label="Analysis contract" role="status">
      <p className="text-[10px] uppercase tracking-wider text-signal">
        Drafting the analysis contract
      </p>
      <p className="mt-1.5 text-[11px] text-ink-dim">
        Stating the claim, choosing tools, and listing what could make the answer
        wrong.
      </p>
      <div className="mt-3 h-0.5 w-full overflow-hidden rounded bg-edge">
        <motion.div
          className="h-full w-1/3 bg-signal"
          animate={{ x: ["-100%", "300%"] }}
          transition={{ duration: 1.3, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>
      {seconds >= PATIENCE_SECONDS && (
        <p className="tabular mt-2.5 text-[10px] leading-snug text-ink-faint">
          Waiting on the language model, {seconds}s. If it does not answer in
          time the offline rule router plans this instead, and the measurements
          are identical either way.
        </p>
      )}
    </section>
  );
}

/**
 * Reveals text a character at a time. The contract is the moment the system says
 * what it thinks it was asked, so it is worth letting the viewer read it.
 * Skips instantly when the user prefers reduced motion.
 */
function Typewriter({
  text,
  speed = 14,
  onDone,
}: {
  text: string;
  speed?: number;
  onDone?: () => void;
}) {
  const reduceMotion = usePrefersReducedMotion();
  const [shown, setShown] = useState(reduceMotion ? text.length : 0);

  // Kept in a ref, not in the effect's dependencies. Completing the reveal makes
  // the parent re-render, which hands back a fresh callback identity; depending
  // on it would tear down the finished interval and retype from zero forever.
  const doneRef = useRef(onDone);
  useEffect(() => {
    doneRef.current = onDone;
  });

  useEffect(() => {
    if (reduceMotion) {
      setShown(text.length);
      doneRef.current?.();
      return;
    }
    setShown(0);
    let index = 0;
    const id = window.setInterval(() => {
      index += 1;
      setShown(index);
      if (index >= text.length) {
        window.clearInterval(id);
        doneRef.current?.();
      }
    }, speed);
    return () => window.clearInterval(id);
  }, [text, speed, reduceMotion]);

  return (
    <span>
      {text.slice(0, shown)}
      {shown < text.length && (
        <span className="animate-[var(--animate-blink)] text-signal">|</span>
      )}
    </span>
  );
}

interface Props {
  contract: AnalysisContract | null;
  loading: boolean;
  error: string | null;
  onApprove: () => void;
  onDiscard: () => void;
  approveLabel?: string;
  approveDisabled?: boolean;
}

/**
 * Routes between the card's four states. The body is remounted per contract
 * hash so its reveal state resets on a new contract without an effect that has
 * to race the reveal itself.
 */
export function ContractCard({
  contract,
  loading,
  error,
  onApprove,
  onDiscard,
  approveLabel = "Approve and run",
  approveDisabled = false,
}: Props) {
  if (loading) {
    return <DraftingCard />;
  }

  if (error) {
    return (
      <section className="panel p-4" aria-label="Analysis contract">
        <p role="alert" className="text-[11px] leading-snug text-disagree">
          {error}
        </p>
      </section>
    );
  }

  if (!contract) return null;

  return (
    <ContractBody
      key={contract.contract_hash}
      contract={contract}
      onApprove={onApprove}
      onDiscard={onDiscard}
      approveLabel={approveLabel}
      approveDisabled={approveDisabled}
    />
  );
}

function ContractBody({
  contract,
  onApprove,
  onDiscard,
  approveLabel,
  approveDisabled,
}: {
  contract: AnalysisContract;
  onApprove: () => void;
  onDiscard: () => void;
  approveLabel: string;
  approveDisabled: boolean;
}) {
  const reduceMotion = usePrefersReducedMotion();
  // With motion off there is no reveal to wait on, so the body opens immediately.
  const [claimDone, setClaimDone] = useState(reduceMotion);
  const [showAudit, setShowAudit] = useState(false);
  const markClaimDone = useCallback(() => setClaimDone(true), []);

  // A backstop on the reveal.
  //
  // Approve is the only way to run anything, and it lives inside the body this
  // gate controls. So a decorative animation is, structurally, holding the
  // application's primary action hostage: if the typewriter's interval is
  // throttled, coalesced in a background tab, or interrupted, the button never
  // arrives and the page looks broken with nothing to click. The reveal is worth
  // keeping and worth exactly none of that risk, so it gets a deadline.
  useEffect(() => {
    if (claimDone) return;
    const grace = Math.max(1200, contract.claim.length * 28);
    const id = window.setTimeout(() => setClaimDone(true), grace);
    return () => window.clearTimeout(id);
  }, [claimDone, contract.claim]);

  const auditCount = contract.repairs.length + contract.rejections.length;

  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="panel panel-raised @container overflow-hidden border-signal/30"
      aria-label="Analysis contract"
    >
      <div className="flex items-center justify-between border-b border-edge px-4 py-2.5">
        <h2 className="text-[10px] uppercase tracking-wider text-signal">
          Analysis contract
        </h2>
        <span className="tabular text-[10px] text-ink-faint">
          {formatDuration(contract.duration_ms)}
        </span>
      </div>

      <div className="px-4 py-3">
        <p className="text-[10px] uppercase tracking-wider text-ink-faint">
          Claim under test
        </p>
        <p className="mt-1 text-sm leading-snug text-ink">
          <Typewriter text={contract.claim} onDone={markClaimDone} />
        </p>
      </div>

      <AnimatePresence>
        {claimDone && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5 border-t border-edge px-4 py-3">
              <dt className="text-[11px] text-ink-faint">Task</dt>
              <dd className="text-[11px] text-ink">
                {TASK_LABEL[contract.task_type]}
                {contract.change_direction !== "unspecified" && (
                  <span className="ml-1.5 text-ink-dim">
                    &middot; asserted {contract.change_direction}
                  </span>
                )}
              </dd>

              {contract.target_classes.length > 0 && (
                <>
                  <dt className="text-[11px] text-ink-faint">Target</dt>
                  <dd className="text-[11px] text-ink">
                    {contract.target_classes.join(", ")}
                  </dd>
                </>
              )}

              <dt className="text-[11px] text-ink-faint">Acts on</dt>
              <dd className="tabular text-[11px] text-ink-dim">
                {contract.acts_on.join(", ")}
              </dd>
            </dl>

            {/*
              Side by side once the card is wide: what will run next to what
              could make it wrong, which is the comparison an approver is making.
              Stacked on a monitor, both lists ran down the left edge of a card a
              screen wide.
            */}
            <div className="@4xl:grid @4xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
              <div className="border-t border-edge px-4 py-3">
                <p className="mb-1.5 text-[10px] uppercase tracking-wider text-ink-faint">
                  Tools to run
                </p>
                <ol className="space-y-1.5">
                  {contract.tools.map((plan, index) => (
                    <motion.li
                      key={plan.tool}
                      initial={{ opacity: 0, x: -6 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.07, duration: 0.3 }}
                      className="flex gap-2"
                    >
                      <span className="tabular mt-0.5 text-[10px] text-signal/70">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <div className="min-w-0">
                        <p className="tabular text-[11px] text-ink">
                          {plan.tool}
                          <span className="ml-1.5 text-ink-faint">v{plan.version}</span>
                        </p>
                        <p className="text-[10px] leading-snug text-ink-faint">
                          {plan.rationale}
                        </p>
                        {Object.keys(plan.parameters).length > 0 && (
                          // A JSON list has no spaces to wrap at, so without
                          // anywhere-wrapping it ran off the card on a phone.
                          <p className="tabular mt-0.5 text-[10px] text-ink-dim [overflow-wrap:anywhere]">
                            {Object.entries(plan.parameters)
                              .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
                              .join("  ")}
                          </p>
                        )}
                      </div>
                    </motion.li>
                  ))}
                </ol>
              </div>

              <div className="border-t border-edge bg-uncertain/5 px-4 py-3 @4xl:border-l">
                <p className="mb-1.5 text-[10px] uppercase tracking-wider text-uncertain">
                  What could make this wrong
                </p>
                <ul className="space-y-1.5">
                  {contract.confounders.map((plan, index) => (
                    <motion.li
                      key={plan.kind}
                      initial={{ opacity: 0, x: -6 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.2 + index * 0.07, duration: 0.3 }}
                    >
                      <p className="text-[11px] text-ink">{plan.label}</p>
                      <p className="text-[10px] leading-snug text-ink-faint">
                        {plan.question}
                        {plan.reason && (
                          <span className="text-ink-dim"> &mdash; {plan.reason}</span>
                        )}
                      </p>
                    </motion.li>
                  ))}
                </ul>
              </div>
            </div>

            {contract.expected_outputs.length > 0 && (
              <div className="border-t border-edge px-4 py-2.5">
                <p className="text-[10px] uppercase tracking-wider text-ink-faint">
                  You will receive
                </p>
                <p className="mt-1 text-[11px] leading-snug text-ink-dim">
                  {contract.expected_outputs.join("; ")}
                </p>
              </div>
            )}

            <div className="border-t border-edge px-4 py-2.5">
              <p className="text-[10px] leading-snug text-ink-faint">
                {SOURCE_LABEL[contract.source]}
                {contract.model && (
                  <span className="tabular text-ink-dim"> ({contract.model})</span>
                )}
                {contract.fallback_reason && contract.source !== "gemini" && (
                  <span className="text-ink-dim"> &mdash; {contract.fallback_reason}</span>
                )}
              </p>
              {contract.interpretation_note && (
                <p className="mt-1 text-[10px] leading-snug text-narrate">
                  {contract.interpretation_note}
                </p>
              )}
            </div>

            {auditCount > 0 && (
              <div className="border-t border-edge">
                <button
                  type="button"
                  onClick={() => setShowAudit((v) => !v)}
                  aria-expanded={showAudit}
                  className="w-full px-4 py-2 text-left text-[10px] uppercase tracking-wider text-ink-faint transition hover:text-ink-dim"
                >
                  {showAudit ? "Hide" : "Show"} {auditCount} validation change
                  {auditCount === 1 ? "" : "s"}
                </button>
                {showAudit && (
                  <div className="space-y-1.5 px-4 pb-3">
                    {contract.rejections.map((rejection) => (
                      <p
                        key={`${rejection.field}-${rejection.value}`}
                        className="text-[10px] leading-snug text-disagree"
                      >
                        Rejected{" "}
                        <span className="tabular">{rejection.value}</span> in{" "}
                        {rejection.field}: {rejection.reason}
                      </p>
                    ))}
                    {contract.repairs.map((repair) => (
                      <p
                        key={`${repair.field}-${repair.detail}`}
                        className="text-[10px] leading-snug text-uncertain"
                      >
                        Adjusted {repair.field}: {repair.detail}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className="flex gap-2 border-t border-edge px-4 py-3">
              <button
                type="button"
                onClick={onApprove}
                disabled={approveDisabled}
                // Full width on a phone, where it is a thumb target. On a wide card a
                // button the width of the screen reads as a banner, not a control.
                className="flex-1 rounded-lg bg-signal px-3 py-2 text-xs font-semibold text-void transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50 @4xl:max-w-xs"
              >
                {approveLabel}
              </button>
              <button
                type="button"
                onClick={onDiscard}
                className="rounded-lg border border-edge px-3 py-2 text-xs text-ink-dim transition hover:border-edge-hi hover:text-ink"
              >
                Discard
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  );
}
