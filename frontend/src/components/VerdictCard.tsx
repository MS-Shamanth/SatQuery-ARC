import { motion } from "framer-motion";
import { useState } from "react";
import { usePrefersReducedMotion } from "../lib/useReducedMotion";
import type {
  ConfidenceComponent,
  ConsistencyCheck,
  EvidenceItem,
  Verdict,
  VerdictLabel,
} from "../lib/types";

const LABEL: Record<
  VerdictLabel,
  { text: string; tone: string; ring: string; bar: string }
> = {
  supported: {
    text: "Supported",
    tone: "text-agree",
    ring: "border-agree/50 bg-agree/10",
    bar: "bg-agree",
  },
  refuted: {
    text: "Refuted",
    tone: "text-disagree",
    ring: "border-disagree/50 bg-disagree/10",
    bar: "bg-disagree",
  },
  inconclusive: {
    text: "Inconclusive",
    tone: "text-uncertain",
    ring: "border-uncertain/50 bg-uncertain/10",
    bar: "bg-uncertain",
  },
  unanswerable: {
    text: "Unanswerable with this data",
    tone: "text-ink-dim",
    ring: "border-edge-hi bg-hull/50",
    bar: "bg-edge-hi",
  },
};

const DIRECTION_ARROW: Record<string, string> = {
  increased: "\u2191",
  decreased: "\u2193",
  unchanged: "\u2192",
  unspecified: "\u00b7",
};

/**
 * The conclusion, with everything it rests on one click away.
 *
 * The confidence figure is drawn as its components rather than as a single bar,
 * because a lone percentage cannot be argued with. Seeing that measurement
 * quality contributed nothing while observability contributed its full weight
 * tells a reader exactly where the finding is weak, and the lever beside it says
 * what would fix it.
 */
export function VerdictCard({ verdict }: { verdict: Verdict }) {
  const reduceMotion = usePrefersReducedMotion();
  const style = LABEL[verdict.label];
  const [showEvidence, setShowEvidence] = useState(false);

  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduceMotion ? 0 : 0.45, ease: [0.16, 1, 0.3, 1] }}
      className={`panel panel-raised overflow-hidden border ${style.ring}`}
      aria-label="Verdict"
    >
      <div className="border-b border-edge px-4 py-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <p className={`text-sm font-semibold ${style.tone}`}>{style.text}</p>
          <p className="tabular text-[11px] text-ink-faint">
            {(verdict.confidence * 100).toFixed(0)}% confidence
          </p>
        </div>
        <p className="mt-1 text-[11px] leading-snug text-ink-dim">
          {verdict.claim}
        </p>

        {verdict.asserted_direction !== "unspecified" && (
          <p className="tabular mt-1.5 text-[10px] text-ink-faint">
            asserted {DIRECTION_ARROW[verdict.asserted_direction]}{" "}
            {verdict.asserted_direction}
            <span className="mx-1.5 text-edge-hi">|</span>
            measured {DIRECTION_ARROW[verdict.measured_direction]}{" "}
            {verdict.measured_direction}
          </p>
        )}
      </div>

      <div className="border-b border-edge px-4 py-3">
        <p className="text-[10px] uppercase tracking-wider text-ink-faint">
          Why
        </p>
        <p className="mt-1 text-[11px] leading-relaxed text-ink">
          {verdict.reasoning}
        </p>

        {verdict.narrative && (
          <div className="mt-2.5 rounded-lg border border-narrate/30 bg-narrate/5 px-3 py-2">
            <p className="text-[9px] uppercase tracking-wider text-narrate">
              Phrased by a language model
              {verdict.narrative_audit && (
                <span className="tabular ml-1.5 text-ink-faint">
                  {verdict.narrative_audit.traced}/
                  {verdict.narrative_audit.checked} figures traced to the ledger
                </span>
              )}
            </p>
            <p className="mt-1 text-[11px] leading-relaxed text-ink-dim">
              {verdict.narrative}
            </p>
          </div>
        )}

        {verdict.narrative_audit && !verdict.narrative_audit.passed && (
          <p className="mt-2.5 rounded-lg border border-disagree/30 bg-disagree/5 px-3 py-2 text-[10px] leading-snug text-disagree">
            {verdict.narrative_audit.note}
          </p>
        )}
      </div>

      <ConfidenceBreakdown components={verdict.confidence_components} />

      {verdict.consistency.length > 0 && (
        <div className="border-b border-edge px-4 py-3">
          <p className="text-[10px] uppercase tracking-wider text-ink-faint">
            Measured twice
          </p>
          <ul className="mt-1.5 space-y-1.5">
            {verdict.consistency.map((check) => (
              <ConsistencyRow key={check.first_key} check={check} />
            ))}
          </ul>
        </div>
      )}

      {verdict.what_would_change_it.length > 0 && (
        <div className="border-b border-edge bg-signal/5 px-4 py-3">
          <p className="text-[10px] uppercase tracking-wider text-signal">
            What would change this answer
          </p>
          <ul className="mt-1.5 space-y-1">
            {verdict.what_would_change_it.map((lever) => (
              <li
                key={lever}
                className="flex gap-2 text-[11px] leading-snug text-ink-dim"
              >
                <span className="text-signal" aria-hidden="true">
                  &rarr;
                </span>
                <span>{lever}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <button
          type="button"
          onClick={() => setShowEvidence((value) => !value)}
          aria-expanded={showEvidence}
          className="w-full px-4 py-2 text-left text-[10px] uppercase tracking-wider text-ink-faint transition hover:text-signal"
        >
          {showEvidence ? "Hide" : "Show"} the {verdict.evidence.length}{" "}
          measurements this rests on
        </button>
        {showEvidence && (
          <ul className="space-y-1 px-4 pb-3">
            {verdict.evidence.map((item) => (
              <EvidenceRow key={item.measurement_key} item={item} />
            ))}
          </ul>
        )}
      </div>
    </motion.section>
  );
}

function ConfidenceBreakdown({
  components,
}: {
  components: ConfidenceComponent[];
}) {
  const reduceMotion = usePrefersReducedMotion();
  if (components.length === 0) return null;

  return (
    <div className="border-b border-edge px-4 py-3">
      <p className="text-[10px] uppercase tracking-wider text-ink-faint">
        Confidence, by component
      </p>
      <ul className="mt-2 space-y-2">
        {components.map((component) => {
          const negative = component.contribution < 0;
          const span = Math.max(component.weight, 0.01);
          const filled = Math.min(1, Math.abs(component.contribution) / span);
          return (
            <li key={component.name}>
              <div className="flex items-baseline justify-between gap-2">
                <p className="text-[11px] text-ink-dim">{component.label}</p>
                <p
                  className={`tabular shrink-0 text-[11px] ${
                    negative ? "text-disagree" : "text-agree"
                  }`}
                >
                  {component.contribution >= 0 ? "+" : ""}
                  {component.contribution.toFixed(2)}
                  <span className="ml-1 text-[9px] text-ink-faint">
                    of {component.weight.toFixed(2)}
                  </span>
                </p>
              </div>
              <div className="mt-1 h-1 w-full overflow-hidden rounded bg-edge">
                <motion.div
                  className={`h-full ${negative ? "bg-disagree" : "bg-agree"}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${filled * 100}%` }}
                  transition={{ duration: reduceMotion ? 0 : 0.5 }}
                />
              </div>
              <p className="mt-1 text-[10px] leading-snug text-ink-faint">
                {component.rationale}
              </p>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function ConsistencyRow({ check }: { check: ConsistencyCheck }) {
  return (
    <li
      className={`border-l-2 pl-2.5 ${
        check.agrees ? "border-agree" : "border-disagree"
      }`}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-2">
        <p className="text-[11px] text-ink">{check.quantity}</p>
        <p className="tabular text-[11px] text-signal">
          {check.first_value.toFixed(3)} vs {check.second_value.toFixed(3)}{" "}
          <span className="text-[9px] text-ink-faint">{check.unit}</span>
        </p>
      </div>
      <p
        className={`tabular text-[10px] ${
          check.agrees ? "text-agree" : "text-disagree"
        }`}
      >
        {(check.relative_difference * 100).toFixed(0)}% apart
        {check.agrees ? " \u2014 agree" : " \u2014 beyond tolerance"}
      </p>
      <p className="mt-0.5 text-[10px] leading-snug text-ink-faint">
        {check.explanation}
      </p>
    </li>
  );
}

function EvidenceRow({ item }: { item: EvidenceItem }) {
  const tone =
    item.direction === "supports"
      ? "border-agree"
      : item.direction === "refutes"
        ? "border-disagree"
        : "border-edge-hi";

  return (
    <li className={`border-l-2 pl-2.5 ${tone}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-2">
        <p className="text-[11px] text-ink-dim">{item.statement}</p>
        <p className="tabular shrink-0 text-[11px] text-signal">
          {item.display}
        </p>
      </div>
      <p className="tabular text-[9px] leading-snug text-ink-faint">
        {item.measurement_key} &middot; {item.source_tool} v{item.source_version}{" "}
        &middot; {item.formula}
      </p>
    </li>
  );
}
