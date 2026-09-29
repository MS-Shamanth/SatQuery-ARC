import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import type { ConfounderTest, ConfounderVerdict } from "../lib/types";

const VERDICT: Record<
  ConfounderVerdict,
  { label: string; tone: string; border: string; dot: string }
> = {
  ruled_out: {
    label: "Ruled out",
    tone: "text-agree",
    border: "border-agree/40 bg-agree/5",
    dot: "bg-agree",
  },
  plausible: {
    label: "Cannot be ruled out",
    tone: "text-uncertain",
    border: "border-uncertain/40 bg-uncertain/5",
    dot: "bg-uncertain",
  },
  likely: {
    label: "Likely explanation",
    tone: "text-disagree",
    border: "border-disagree/50 bg-disagree/10",
    dot: "bg-disagree",
  },
  not_tested: {
    label: "Not tested",
    tone: "text-ink-faint",
    border: "border-dashed border-edge-hi",
    dot: "bg-edge-hi",
  },
};

/**
 * What could make the answer wrong, and what the system did about it.
 *
 * The ordering is deliberate: surviving explanations come first. A viewer
 * skimming this panel should see the objections that are still standing before
 * the ones that were cleared, because those are what qualify the finding.
 */
export function ConfounderPanel({ tests }: { tests: ConfounderTest[] }) {
  // Defaulted rather than dotted into directly. An array is never dropped from
  // the wire the way a null is, but a panel that reads `.length` off whatever it
  // was handed is one bad payload away from taking the page down.
  if (!tests?.length) return null;

  const ranking: Record<ConfounderVerdict, number> = {
    likely: 0,
    plausible: 1,
    not_tested: 2,
    ruled_out: 3,
  };
  const ordered = [...tests].sort(
    (a, b) => ranking[a.verdict] - ranking[b.verdict],
  );
  const cleared = tests.filter((test) => test.verdict === "ruled_out").length;
  const standing = tests.length - cleared;
  const requirements = [
    ...new Set(
      tests
        .map((test) => test.requirement)
        .filter((value): value is string => Boolean(value)),
    ),
  ];

  return (
    <section className="panel @container overflow-hidden" aria-label="Confounder tests">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge px-4 py-2.5">
        <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
          What could make this wrong
        </h2>
        <p className="tabular text-[10px] text-ink-faint">
          <span className="text-agree">{cleared} ruled out</span>
          {standing > 0 && (
            <>
              {" \u00b7 "}
              <span className="text-uncertain">{standing} still standing</span>
            </>
          )}
        </p>
      </div>

      {/*
        Two to a row on a wide panel, so a test's name and its verdict are not a
        screen's width apart. Row-major, so the explanations still standing, which
        sort first, fill the top row. Each cell's verdict stripe doubles as the
        line between the columns, and the list tucks a pixel under whatever edge
        follows it so the last row does not draw a second one.
      */}
      <ul className="divide-y divide-edge @4xl:-mb-px @4xl:grid @4xl:grid-cols-2">
        {ordered.map((test) => (
          <TestRow key={test.kind} test={test} />
        ))}
      </ul>

      {requirements.length > 0 && (
        // Relative so it paints above the grid cells it overlaps by a pixel. Grid
        // items paint after plain blocks, so without it the last row's coloured
        // underline showed through where this block's own edge should be.
        <div className="relative border-t border-edge bg-signal/5 px-4 py-3">
          <p className="text-[10px] uppercase tracking-wider text-signal">
            What would settle it
          </p>
          <ul className="mt-1.5 space-y-1">
            {requirements.map((requirement) => (
              <li
                key={requirement}
                className="flex gap-2 text-[11px] leading-snug text-ink-dim"
              >
                <span className="text-signal" aria-hidden="true">
                  &rarr;
                </span>
                <span>{requirement}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function TestRow({ test }: { test: ConfounderTest }) {
  const [open, setOpen] = useState(false);
  const style = VERDICT[test.verdict];

  return (
    <li className={`border-l-2 ${style.border} @4xl:border-b`}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-start gap-2.5 px-3.5 py-2.5 text-left transition hover:bg-panel-hi/40"
      >
        <span
          className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${style.dot}`}
          aria-hidden="true"
        />
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-baseline justify-between gap-x-3">
            <span className="text-[11px] font-medium text-ink">{test.label}</span>
            <span className={`text-[10px] uppercase tracking-wide ${style.tone}`}>
              {style.label}
            </span>
          </span>
          <span className="mt-0.5 block text-[10px] leading-snug text-ink-faint">
            {test.question}
          </span>
          <span className="tabular mt-1 block text-[11px] text-signal">
            {test.measured}
          </span>
        </span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <dl className="space-y-2 px-3.5 pb-3 pl-8">
              <Detail label="What this means" value={test.explanation} />
              {test.threshold && (
                <Detail label="Threshold" value={test.threshold} />
              )}
              {test.formula && (
                <Detail label="Formula" value={test.formula} mono />
              )}
              {test.method && <Detail label="Method" value={test.method} />}
              {Object.keys(test.inputs).length > 0 && (
                <Detail
                  label="Inputs"
                  mono
                  value={Object.entries(test.inputs)
                    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
                    .join("  ")}
                />
              )}
            </dl>
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  );
}

function Detail({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <dt className="text-[9px] uppercase tracking-wider text-ink-faint">
        {label}
      </dt>
      <dd
        className={`text-[10px] leading-snug text-ink-dim ${mono ? "tabular" : ""}`}
      >
        {value}
      </dd>
    </div>
  );
}
