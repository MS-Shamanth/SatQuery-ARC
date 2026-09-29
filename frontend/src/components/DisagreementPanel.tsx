import { motion } from "framer-motion";
import { formatArea } from "../lib/format";
import { usePrefersReducedMotion } from "../lib/useReducedMotion";
import type { MapLayer, Measurement } from "../lib/types";

const AGREE_KEY = "fusion_agree_water";
const SAR_ONLY_KEY = "fusion_disagree_sar_only";
const OPTICAL_ONLY_KEY = "fusion_disagree_optical_only";
const BLIND_KEY = "fusion_sar_under_cloud";

/** Above this the two instruments are telling the same story. */
const STRONG_IOU = 0.6;
const FAIR_IOU = 0.35;

function find(measurements: Measurement[], key: string): Measurement | undefined {
  return measurements.find((item) => item.key === key);
}

/**
 * The optical and radar debate, scored.
 *
 * Both agreement figures are shown because they answer different questions. Raw
 * overlap says how much of the water the two instruments placed in the same spot.
 * Cohen's kappa asks how much of that agreement survives once you account for the
 * fact that both would correctly call most of a scene dry by default. On a
 * mostly-dry scene the first flatters and the second does not.
 */
export function DisagreementPanel({
  measurements,
  layers,
  notes,
}: {
  measurements: Measurement[];
  layers: MapLayer[];
  notes: string[];
}) {
  const iou = find(measurements, "agreement_iou");
  const kappa = find(measurements, "agreement_kappa");
  if (!iou || !kappa) return null;

  const agreed = find(measurements, "agreed_water_km2");
  const sarOnly = find(measurements, "sar_only_km2");
  const opticalOnly = find(measurements, "optical_only_km2");
  const cloudShare = find(measurements, "sar_only_under_cloud_share");

  const strength =
    iou.value >= STRONG_IOU
      ? { text: "Strong agreement", tone: "text-agree" }
      : iou.value >= FAIR_IOU
        ? { text: "Partial agreement", tone: "text-uncertain" }
        : { text: "They disagree", tone: "text-disagree" };

  const bands = [
    {
      key: AGREE_KEY,
      label: "Both sensors",
      value: agreed?.value ?? null,
      colour: "var(--color-agree)",
      detail: "Reflectance and backscatter independently place water here.",
    },
    {
      key: SAR_ONLY_KEY,
      label: "Radar only",
      value: sarOnly?.value ?? null,
      colour: "var(--color-uncertain)",
      detail: "Radar sees water the optical result does not.",
    },
    {
      key: OPTICAL_ONLY_KEY,
      label: "Optical only",
      value: opticalOnly?.value ?? null,
      colour: "var(--color-disagree)",
      detail: "The optical result sees water radar does not.",
    },
  ];
  const total = bands.reduce((sum, band) => sum + (band.value ?? 0), 0);
  const blind = layers.find((layer) => layer.key === BLIND_KEY);
  const relevant = notes.filter(
    (note) =>
      note.includes("optical cloud") ||
      note.includes("unrelated physics") ||
      note.includes("could not see") ||
      note.includes("kappa"),
  );

  return (
    <section className="panel @container overflow-hidden" aria-label="Sensor agreement">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge px-4 py-2.5">
        <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
          Optical against radar
        </h2>
        <p className={`text-[11px] font-medium ${strength.tone}`}>
          {strength.text}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-px bg-edge">
        <Figure
          label="Overlap"
          value={iou.value.toFixed(2)}
          note="intersection over union"
        />
        <Figure
          label="Kappa"
          value={kappa.value.toFixed(2)}
          note={String(kappa.inputs.interpretation ?? "corrected for chance")}
        />
      </div>

      <div className="px-4 py-3">
        <p className="text-[10px] uppercase tracking-wider text-ink-faint">
          Where they agree and differ
        </p>

        <div
          className="mt-2 flex h-2.5 w-full overflow-hidden rounded bg-edge"
          role="img"
          aria-label="Proportion of water each sensor found"
        >
          {bands.map((band) => (
            <Segment
              key={band.key}
              colour={band.colour}
              share={total > 0 ? (band.value ?? 0) / total : 0}
              label={band.label}
            />
          ))}
        </div>

        {/*
          Three across once the panel is wide, each area beside its label. As a
          list on a monitor, a band's name and its area were a screen apart.
        */}
        <ul className="mt-2.5 grid grid-cols-1 gap-y-1.5 @3xl:grid-cols-3 @3xl:gap-x-6">
          {bands.map((band) => (
            <li key={band.key} className="flex items-start gap-2.5">
              <span
                className="mt-1 h-2 w-2 shrink-0 rounded-sm"
                style={{ backgroundColor: band.colour }}
                aria-hidden="true"
              />
              <span className="min-w-0 flex-1">
                <span className="flex items-baseline justify-between gap-2 @3xl:justify-start">
                  <span className="text-[11px] text-ink">{band.label}</span>
                  <span className="tabular shrink-0 text-[11px] text-signal">
                    {formatArea(band.value)}
                  </span>
                </span>
                <span className="block text-[10px] leading-snug text-ink-faint">
                  {band.detail}
                </span>
              </span>
            </li>
          ))}
        </ul>
      </div>

      {blind && (
        <div className="border-t border-edge bg-signal/5 px-4 py-2.5">
          <div className="flex items-baseline justify-between gap-2 @3xl:justify-start">
            <p className="text-[11px] text-signal">
              Radar water where optical could not see
            </p>
            <p className="tabular text-[11px] text-signal">
              {formatArea(blind.area_km2)}
            </p>
          </div>
          <p className="mt-0.5 text-[10px] leading-snug text-ink-faint">
            Excluded from the agreement figures. A sensor that made no claim
            cannot be said to disagree.
          </p>
        </div>
      )}

      {cloudShare && (
        <div className="border-t border-edge px-4 py-2.5">
          <div className="flex items-baseline justify-between gap-2 @3xl:justify-start">
            <p className="text-[11px] text-ink-dim">
              Radar-only water sitting under optical cloud
            </p>
            <p className="tabular text-[11px] text-signal">
              {(cloudShare.value * 100).toFixed(0)}%
            </p>
          </div>
        </div>
      )}

      {relevant.length > 0 && (
        <div className="border-t border-edge px-4 py-2.5">
          {relevant.map((note) => (
            <p
              key={note}
              className="mt-1 text-[10px] leading-relaxed text-ink-faint first:mt-0"
            >
              {note}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}

function Figure({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="bg-panel px-4 py-3">
      <p className="text-[10px] uppercase tracking-wider text-ink-faint">
        {label}
      </p>
      <p className="tabular mt-0.5 text-xl text-signal">{value}</p>
      <p className="text-[10px] leading-snug text-ink-faint">{note}</p>
    </div>
  );
}

function Segment({
  colour,
  share,
  label,
}: {
  colour: string;
  share: number;
  label: string;
}) {
  const reduceMotion = usePrefersReducedMotion();
  if (share <= 0) return null;
  return (
    <motion.span
      className="h-full"
      style={{ backgroundColor: colour }}
      initial={{ width: 0 }}
      animate={{ width: `${share * 100}%` }}
      transition={{ duration: reduceMotion ? 0 : 0.5 }}
      title={`${label}: ${(share * 100).toFixed(0)}%`}
    />
  );
}
