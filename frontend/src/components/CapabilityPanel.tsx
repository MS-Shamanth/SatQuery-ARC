import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import type {
  SessionToolAvailability,
  ToolDescriptor,
  ToolImplementation,
} from "../lib/types";

/**
 * What the system can do, and what it cannot do on this particular input.
 *
 * Showing the unavailable tools with their reasons is the point. "NDBI needs
 * SWIR, which this image does not provide" is a more trustworthy thing to read
 * than a tool list that silently shrinks.
 */
const IMPLEMENTATION_LABEL: Record<ToolImplementation, string> = {
  deterministic: "computed",
  learned: "trained model",
  "llm-narration": "wording only",
};

const IMPLEMENTATION_STYLE: Record<ToolImplementation, string> = {
  deterministic: "text-signal",
  learned: "text-agree",
  "llm-narration": "text-narrate",
};

interface Props {
  tools: ToolDescriptor[];
  availability: SessionToolAvailability | null;
  loading: boolean;
  error: string | null;
}

export function CapabilityPanel({ tools, availability, loading, error }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);

  const availableSet = new Set(availability?.available ?? []);
  const reasons = new Map(
    (availability?.unavailable ?? []).map((item) => [item.tool, item.reason]),
  );
  const declined = Object.entries(availability?.declined_registration ?? {});

  return (
    <section className="panel p-4" aria-label="Specialist tools">
      <div className="flex items-baseline justify-between">
        <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
          Specialist tools
        </h2>
        {availability && (
          <span className="tabular text-[10px] text-ink-faint">
            {availableSet.size} of {tools.length} usable here
          </span>
        )}
      </div>

      {loading && (
        <p className="mt-2 text-[11px] text-ink-faint">Reading the registry…</p>
      )}
      {error && (
        <p role="alert" className="mt-2 text-[11px] text-disagree">
          {error}
        </p>
      )}

      <ul className="mt-2.5 space-y-1.5">
        {tools.map((tool) => {
          const usable = availability ? availableSet.has(tool.name) : true;
          const reason = reasons.get(tool.name);
          const open = expanded === tool.name;
          return (
            <li key={tool.name}>
              <button
                type="button"
                onClick={() => setExpanded(open ? null : tool.name)}
                aria-expanded={open}
                className={`w-full rounded-lg border px-2.5 py-2 text-left transition ${
                  usable
                    ? "border-edge bg-hull/40 hover:border-edge-hi"
                    : "border-edge/50 bg-hull/20"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                      usable ? "bg-agree" : "bg-ink-faint/60"
                    }`}
                  />
                  <span
                    className={`tabular min-w-0 flex-1 truncate text-[11px] ${
                      usable ? "text-ink" : "text-ink-faint"
                    }`}
                  >
                    {tool.name}
                  </span>
                  <span
                    className={`shrink-0 text-[9px] uppercase tracking-wide ${
                      IMPLEMENTATION_STYLE[tool.implementation]
                    }`}
                  >
                    {IMPLEMENTATION_LABEL[tool.implementation]}
                  </span>
                </div>
                {!usable && reason && (
                  <p className="mt-1 pl-3.5 text-[10px] leading-snug text-uncertain">
                    {reason}
                  </p>
                )}
              </button>

              <AnimatePresence initial={false}>
                {open && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.18 }}
                    className="overflow-hidden"
                  >
                    <div className="space-y-1 px-2.5 pb-2 pt-1.5">
                      <p className="text-[10px] leading-snug text-ink-dim">
                        {tool.summary}
                      </p>
                      <p className="tabular text-[10px] text-ink-faint">
                        version {tool.version}
                      </p>
                      {tool.requirement.description && (
                        <p className="text-[10px] leading-snug text-ink-faint">
                          Needs: {tool.requirement.description}
                        </p>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </li>
          );
        })}
      </ul>

      {declined.length > 0 && (
        <div className="mt-2.5 border-t border-edge pt-2">
          <p className="text-[10px] uppercase tracking-wider text-ink-faint">
            Not loaded
          </p>
          <ul className="mt-1 space-y-0.5">
            {declined.map(([name, reason]) => (
              <li key={name} className="text-[10px] leading-snug text-ink-faint">
                <span className="tabular text-ink-dim">{name}</span>: {reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="mt-2.5 text-[10px] leading-snug text-ink-faint">
        Measurements come from the computed tools. The language model interprets
        the question and words the explanation; it never produces a number.
      </p>
    </section>
  );
}
