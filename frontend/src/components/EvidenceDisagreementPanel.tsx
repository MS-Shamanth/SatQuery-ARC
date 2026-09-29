import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { formatArea } from "../lib/format";
import { usePrefersReducedMotion } from "../lib/useReducedMotion";
import type {
  Agreement,
  ConflictRegion,
  DisagreementReport,
  MapLayer,
  MethodOpinion,
} from "../lib/types";

const TONE: Record<
  Agreement,
  { label: string; text: string; border: string; bg: string; swatch: string }
> = {
  agree: {
    label: "Methods agree",
    text: "text-agree",
    border: "border-agree/40",
    bg: "bg-agree/10",
    swatch: "bg-agree",
  },
  disagree: {
    label: "Methods disagree",
    text: "text-disagree",
    border: "border-disagree/40",
    bg: "bg-disagree/10",
    swatch: "bg-disagree",
  },
  uncertain: {
    label: "Not enough confidence to say",
    text: "text-uncertain",
    border: "border-uncertain/40",
    bg: "bg-uncertain/10",
    swatch: "bg-uncertain",
  },
};

/**
 * Where the evidence conflicts, as the headline rather than a footnote.
 *
 * Every other panel reports what was found. This one reports where the findings
 * contradict each other, which is more useful and much harder to fake. A single
 * mask on a map invites belief; ground marked as contested tells the reader
 * exactly where belief would be misplaced.
 *
 * The red is not an error display. Each method ran correctly and reported what it
 * measures, and they differ because the ground is ambiguous at this resolution or
 * because the methods measure different physical quantities. So no winner is
 * declared anywhere in this component: there is no labelled reference for these
 * scenes, and choosing one instrument over another would be inventing the ground
 * truth the rest of the system refuses to invent. The recommended action is always
 * review by a person.
 */
export function EvidenceDisagreementPanel({
  disagreement,
  layers = [],
}: {
  disagreement: DisagreementReport | null;
  layers?: MapLayer[];
}) {
  const reduceMotion = usePrefersReducedMotion();
  const [open, setOpen] = useState<number | null>(null);

  // Nullish, not strictly null.
  //
  // A trace that has not reached the comparison stage yet may not carry this key
  // at all, and `=== null` waved the undefined through to the destructure on the
  // next line. The panel then took itself down with "Cannot destructure property
  // 'candidate_area_km2'" over a run that was working perfectly.
  //
  // The server no longer drops nulls from the stream, which was the cause. This is
  // the belt to that braces: a panel should render nothing when it has nothing,
  // never throw, whatever shape the absence arrives in.
  if (disagreement == null) return null;

  const { candidate_area_km2: candidate } = disagreement;
  const share = (value: number) =>
    candidate > 0 ? Math.max(0, Math.min(1, value / candidate)) : 0;

  const bars: { key: Agreement; area: number }[] = [
    { key: "agree", area: disagreement.agree_area_km2 },
    { key: "disagree", area: disagreement.disagree_area_km2 },
    { key: "uncertain", area: disagreement.uncertain_area_km2 },
  ];

  const contested = disagreement.regions.filter(
    (region) => region.agreement !== "agree",
  );

  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduceMotion ? 0 : 0.45, ease: [0.16, 1, 0.3, 1] }}
      className="panel panel-raised @container overflow-hidden border border-edge-hi"
      aria-label="Evidence disagreement"
    >
      <div className="border-b border-edge px-4 py-3">
        <p className="text-[10px] uppercase tracking-wider text-ink-faint">
          Evidence disagreement
        </p>
        <p className="mt-1.5 text-2xl leading-none text-disagree">
          <span className="tabular font-semibold">
            {(disagreement.conflicting_fraction * 100).toFixed(1)}%
          </span>
        </p>
        <p className="mt-1 text-[11px] leading-snug text-ink-dim">
          of the candidate area has conflicting evidence across{" "}
          {disagreement.methods.length} methods. These are not errors. They are
          the ground where the methods do not agree with each other.
        </p>
      </div>

      {/* A single stacked bar, because the three quantities are parts of one whole. */}
      <div className="border-b border-edge px-4 py-3">
        <div
          className="flex h-3 w-full overflow-hidden rounded bg-edge"
          role="img"
          aria-label="Agreement across the candidate area"
        >
          {bars.map(({ key, area }) => (
            <motion.div
              key={key}
              className={TONE[key].swatch}
              initial={{ width: 0 }}
              animate={{ width: `${share(area) * 100}%` }}
              transition={{ duration: reduceMotion ? 0 : 0.6 }}
              title={`${TONE[key].label}: ${formatArea(area)}`}
            />
          ))}
        </div>

        <ul className="mt-2.5 grid gap-1.5 sm:grid-cols-3">
          {bars.map(({ key, area }) => (
            <li key={key} className="flex items-baseline gap-2">
              <span
                className={`mt-1 h-2 w-2 shrink-0 rounded-sm ${TONE[key].swatch}`}
                aria-hidden="true"
              />
              <span className="min-w-0">
                <span className={`block text-[10px] ${TONE[key].text}`}>
                  {TONE[key].label}
                </span>
                <span className="tabular block text-[11px] text-ink-dim">
                  {formatArea(area)}
                </span>
              </span>
            </li>
          ))}
        </ul>

        <p className="tabular mt-2 text-[10px] text-ink-faint">
          Candidate area {formatArea(candidate)} &middot; compared:{" "}
          {disagreement.methods.join(", ")}
        </p>
      </div>

      {contested.length > 0 && (
        <div className="border-b border-edge">
          <p className="px-4 pt-3 text-[10px] uppercase tracking-wider text-ink-faint">
            {contested.length} contested region
            {contested.length === 1 ? "" : "s"}, largest first
          </p>
          {/*
            Two or three to a row once the panel is wide. As one list on a monitor,
            a run's thirty-odd regions were a column of rows each a screen wide,
            with a region's name at one edge and its area at the other. Row-major,
            so the top row still holds the largest regions and the list reads in
            rank order.

            In the grid each cell draws its own right and bottom edge, and the list
            is pulled a pixel under the block's bottom border and the panel's
            clipped right side. The outermost edges then land on lines that are
            already there instead of doubling them.
          */}
          <ul className="mt-1.5 divide-y divide-edge @4xl:-mr-px @4xl:-mb-px @4xl:grid @4xl:grid-cols-2 @7xl:grid-cols-3">
            {contested.map((region) => (
              <RegionRow
                key={region.region_id}
                region={region}
                expanded={open === region.region_id}
                onToggle={() =>
                  setOpen(open === region.region_id ? null : region.region_id)
                }
              />
            ))}
          </ul>
        </div>
      )}

      {layers.some((layer) => layer.key.startsWith("evidence_")) && (
        <p className="border-b border-edge px-4 py-2.5 text-[10px] leading-snug text-ink-faint">
          The same three states are drawn on the map above as
          <span className="text-agree"> agree</span>,
          <span className="text-disagree"> disagree</span> and
          <span className="text-uncertain"> uncertain</span> overlays, counted from
          the arrays these figures came from.
        </p>
      )}

      {disagreement.notes.length > 0 && (
        <ul className="space-y-1 px-4 py-3">
          {disagreement.notes.map((note) => (
            <li
              key={note}
              className="flex gap-2 text-[10px] leading-snug text-ink-faint"
            >
              <span className="text-ink-faint/60" aria-hidden="true">
                &middot;
              </span>
              <span>{note}</span>
            </li>
          ))}
        </ul>
      )}
    </motion.section>
  );
}

function RegionRow({
  region,
  expanded,
  onToggle,
}: {
  region: ConflictRegion;
  expanded: boolean;
  onToggle: () => void;
}) {
  const reduceMotion = usePrefersReducedMotion();
  const tone = TONE[region.agreement];
  const [lon, lat] = region.centroid_wgs84;

  return (
    <li className="@4xl:border-r @4xl:border-b @4xl:border-edge">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className="flex w-full items-baseline justify-between gap-2 px-4 py-2.5 text-left transition hover:bg-panel-hi/40"
      >
        <span className="flex min-w-0 items-baseline gap-2">
          <span
            className={`mt-1 h-2 w-2 shrink-0 rounded-sm ${tone.swatch}`}
            aria-hidden="true"
          />
          <span className="tabular text-[11px] text-ink">
            Region #{region.region_id}
          </span>
          <span className={`text-[10px] ${tone.text}`}>{tone.label}</span>
        </span>
        <span className="tabular shrink-0 text-[11px] text-signal">
          {formatArea(region.area_km2)}
        </span>
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: reduceMotion ? 0 : 0.25 }}
            className="overflow-hidden"
          >
            <div className={`mx-4 mb-3 rounded-lg border ${tone.border} ${tone.bg} px-3 py-2.5`}>
              {region.centroid_wgs84.length === 2 && (
                <p className="tabular text-[10px] text-ink-faint">
                  {lat.toFixed(4)}N {lon.toFixed(4)}E &middot;{" "}
                  {region.pixel_count.toLocaleString("en-IN")} px
                </p>
              )}

              <ul className="mt-2 space-y-1">
                {region.opinions.map((opinion) => (
                  <OpinionRow
                    key={`${opinion.method}-${opinion.verdict}`}
                    opinion={opinion}
                  />
                ))}
              </ul>

              <div className="mt-2.5 border-t border-edge pt-2">
                <p className="text-[9px] uppercase tracking-wider text-ink-faint">
                  Reason
                </p>
                <p className="mt-0.5 text-[11px] leading-snug text-ink-dim">
                  {region.reason}
                </p>
                {region.reason_measurement_key && (
                  <p className="tabular mt-0.5 text-[9px] text-ink-faint">
                    from {region.reason_measurement_key}
                    {region.reason_value !== null &&
                      ` = ${region.reason_value.toFixed(3)}`}
                  </p>
                )}
              </div>

              <div className="mt-2 border-t border-edge pt-2">
                <p className="text-[9px] uppercase tracking-wider text-ink-faint">
                  Action
                </p>
                <p className={`mt-0.5 text-[11px] font-semibold ${tone.text}`}>
                  {region.action}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  );
}

function OpinionRow({ opinion }: { opinion: MethodOpinion }) {
  // A method that could not see the ground is shown as abstaining, not dissenting.
  const dissent = !opinion.could_not_see && opinion.verdict === opinion.verdict.toUpperCase();

  return (
    <li className="flex flex-wrap items-baseline justify-between gap-x-2">
      <span className="text-[11px] text-ink-dim">{opinion.method}</span>
      <span
        className={`tabular text-[11px] ${
          opinion.could_not_see
            ? "text-ink-faint"
            : dissent
              ? "text-signal"
              : "text-ink-faint"
        }`}
      >
        &rarr; {opinion.verdict}
      </span>
      <span className="tabular w-full text-[9px] leading-snug text-ink-faint/80">
        {opinion.source_tool} v{opinion.source_version}
        {opinion.confidence !== null &&
          ` \u00b7 separability ${opinion.confidence.toFixed(2)}`}
      </span>
    </li>
  );
}
