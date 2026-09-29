import { motion } from "framer-motion";
import { useState } from "react";
import { usePrefersReducedMotion } from "../lib/useReducedMotion";
import type { RemedyOffer, RemedySet } from "../lib/types";

interface Props {
  remedies: RemedySet | null;
  /** Loads the offered scene and asks its question. */
  onTake: (offer: RemedyOffer) => void | Promise<void>;
  busyKey?: string | null;
  disabled?: boolean;
}

/**
 * The refusal, made actionable.
 *
 * An inconclusive answer names the imagery that would settle it. When the sample
 * library already contains that imagery, printing the requirement and stopping
 * there wastes the fact. These buttons load the scene that removes the objection
 * and re-ask the same question, so the verdict changes because the confounder is
 * gone rather than because a threshold was nudged.
 *
 * Objections nothing in the library can answer are listed too. An empty offer
 * list must not read as an empty objection list.
 */
export function RemedyOffers({ remedies, onTake, busyKey, disabled }: Props) {
  const reduceMotion = usePrefersReducedMotion();
  const [taken, setTaken] = useState<string | null>(null);

  if (!remedies) return null;
  if (remedies.offers.length === 0 && remedies.unmet.length === 0) return null;

  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduceMotion ? 0 : 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="panel panel-raised border border-signal/40 bg-signal/5"
      aria-label="Imagery that would settle this"
    >
      <div className="border-b border-edge px-4 py-3">
        <p className="text-[10px] uppercase tracking-wider text-signal">
          Imagery that would settle this
        </p>
        <p className="mt-1 text-[11px] leading-snug text-ink-dim">
          {remedies.offers.length > 0
            ? "These scenes remove the objection by construction. Loading one re-asks the same question against data the confounder cannot explain."
            : "Nothing in the sample library removes what is still standing."}
        </p>
      </div>

      {remedies.offers.length > 0 && (
        <ul className="divide-y divide-edge">
          {remedies.offers.map((offer) => {
            const key = `${offer.confounder}:${offer.sample_key}`;
            const busy = busyKey === offer.sample_key;
            return (
              <li key={key} className="px-4 py-3">
                <div className="flex flex-wrap items-baseline justify-between gap-x-2 gap-y-1">
                  <p className="text-[11px] font-semibold text-ink">
                    {offer.title}
                  </p>
                  {offer.corrects_current && (
                    <span className="rounded border border-signal/40 bg-signal/10 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-signal">
                      built for this objection
                    </span>
                  )}
                </div>

                <p className="tabular mt-0.5 text-[10px] text-ink-faint">
                  {offer.place}
                  <span className="mx-1.5 text-edge-hi">|</span>
                  {offer.sample_key}
                </p>

                <p className="mt-1.5 text-[10px] leading-snug text-ink-faint">
                  <span className="text-uncertain">
                    {offer.confounder_label} still standing.
                  </span>{" "}
                  Needs: {offer.requirement}
                </p>

                <p className="mt-1 text-[11px] leading-snug text-ink-dim">
                  {offer.why}
                </p>

                <button
                  type="button"
                  disabled={disabled || busy}
                  onClick={() => {
                    setTaken(key);
                    void onTake(offer);
                  }}
                  className="mt-2 rounded-lg bg-signal px-3 py-1.5 text-[11px] font-semibold text-void transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {busy
                    ? "Loading\u2026"
                    : taken === key
                      ? "Loaded \u2014 re-asking"
                      : "Load it and re-ask"}
                </button>

                {offer.suggested_query && (
                  <p className="mt-1.5 text-[10px] leading-snug text-ink-faint">
                    Will ask: &ldquo;{offer.suggested_query}&rdquo;
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {remedies.unmet.length > 0 && (
        <div className="border-t border-edge px-4 py-3">
          <p className="text-[10px] uppercase tracking-wider text-ink-faint">
            Still unanswered by anything on hand
          </p>
          <ul className="mt-1.5 space-y-1">
            {remedies.unmet.map((requirement) => (
              <li
                key={requirement}
                className="flex gap-2 text-[10px] leading-snug text-ink-faint"
              >
                <span className="text-uncertain" aria-hidden="true">
                  &middot;
                </span>
                <span>{requirement}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </motion.section>
  );
}
