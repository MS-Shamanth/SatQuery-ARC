import { motion } from "framer-motion";
import { useState } from "react";
import { BandChips } from "./BandChips";
import {
  NOT_MEASURED,
  formatArea,
  formatBytes,
  formatDate,
  formatDimensions,
  formatDuration,
  formatGsd,
  formatLatLon,
  formatPercent,
} from "../lib/format";
import type { IngestedImage } from "../lib/types";

const MODALITY_STYLE = {
  optical: "border-signal/40 bg-signal/10 text-signal",
  sar: "border-uncertain/40 bg-uncertain/10 text-uncertain",
  unknown: "border-edge-hi bg-panel-hi text-ink-dim",
} as const;

function Row({
  label,
  value,
  hint,
  mono = true,
}: {
  label: string;
  value: string;
  hint?: string;
  mono?: boolean;
}) {
  const measured = value !== NOT_MEASURED;
  return (
    <>
      <dt className="text-[11px] text-ink-faint">{label}</dt>
      <dd
        className={`text-[11px] ${mono ? "tabular" : ""} ${
          measured ? "text-ink" : "text-ink-faint italic"
        }`}
        title={hint}
      >
        {value}
        {hint && measured && (
          <span className="ml-1.5 not-italic text-ink-faint">{hint}</span>
        )}
      </dd>
    </>
  );
}

export function MetadataPanel({
  image,
  thumbnailUrl,
  onRemove,
}: {
  image: IngestedImage;
  thumbnailUrl: string;
  onRemove?: () => void;
}) {
  const [showTags, setShowTags] = useState(false);
  const { metadata: meta } = image;
  const geo = meta.geo;
  const tagEntries = Object.entries(meta.tags);

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
      aria-label={`Measured metadata for ${meta.original_filename}`}
      // A size container: the card lays itself out by its own width, because two
      // cards side by side on a wide screen are as narrow as one on a tablet.
      className="panel panel-raised @container overflow-hidden"
    >
      <div className="flex items-start justify-between gap-3 border-b border-edge px-4 py-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-ink">
            {meta.original_filename}
          </p>
          <p className="tabular mt-0.5 text-[11px] text-ink-faint">
            {meta.driver} &middot; {formatBytes(meta.size_bytes)} &middot; read in{" "}
            {formatDuration(image.ingest_ms)}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span
            className={`rounded-md border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide ${
              MODALITY_STYLE[meta.modality.modality]
            }`}
            title={meta.modality.reasons.join("; ")}
          >
            {meta.modality.modality}
          </span>
          {onRemove && (
            <button
              type="button"
              onClick={onRemove}
              aria-label={`Remove ${meta.original_filename}`}
              className="rounded-md border border-edge px-2 py-0.5 text-[10px] text-ink-faint transition hover:border-disagree/50 hover:text-disagree"
            >
              Remove
            </button>
          )}
        </div>
      </div>

      {/* Preview on top in a narrow card, beside the readings once there is room. */}
      <div className="grid gap-4 p-4 @min-[26rem]:grid-cols-[minmax(0,150px)_minmax(0,1fr)] @2xl:grid-cols-[minmax(0,200px)_minmax(0,1fr)] @4xl:grid-cols-[minmax(0,240px)_minmax(0,1fr)]">
        <div>
          <div className="overflow-hidden rounded-lg border border-edge bg-abyss">
            <img
              src={thumbnailUrl}
              alt={`Preview of ${meta.original_filename}`}
              className="block h-auto w-full"
              loading="lazy"
            />
          </div>
          {image.thumbnail_recipe && (
            <p className="mt-1.5 text-[10px] leading-snug text-ink-faint">
              {image.thumbnail_recipe}
            </p>
          )}
        </div>

        <div>
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5">
            <Row label="Dimensions" value={formatDimensions(meta.width, meta.height)} />
            <Row
              label="Bands"
              value={`${meta.band_count} \u00d7 ${meta.dtype}`}
            />
            <Row
              label="CRS"
              value={geo.epsg ? `EPSG:${geo.epsg}` : NOT_MEASURED}
              hint={geo.crs_name ?? undefined}
              mono
            />
            <Row
              label="Resolution"
              value={formatGsd(geo.gsd_m)}
              hint={
                geo.gsd_method === "geodesic-at-centre"
                  ? "geodesic at scene centre"
                  : geo.gsd_method === "projected-linear-unit"
                    ? "from affine transform"
                    : undefined
              }
            />
            <Row label="Footprint" value={formatArea(geo.area_km2)} />
            <Row label="Centre" value={formatLatLon(geo.centroid_wgs84)} />
            <Row
              label="Acquired"
              value={formatDate(meta.acquisition_date)}
              hint={
                meta.acquisition_date
                  ? `${meta.date_source}${
                      meta.day_of_year ? `, DOY ${meta.day_of_year}` : ""
                    }`
                  : undefined
              }
            />
            <Row label="NoData" value={formatPercent(meta.nodata_fraction)} />
          </dl>

          <div className="mt-3.5">
            <p className="mb-1.5 text-[10px] uppercase tracking-wider text-ink-faint">
              Bands resolved
            </p>
            <BandChips bands={meta.bands} />
          </div>

          <div className="mt-3">
            <p className="mb-1 text-[10px] uppercase tracking-wider text-ink-faint">
              Modality inferred because
            </p>
            <ul className="space-y-0.5">
              {meta.modality.reasons.map((reason) => (
                <li key={reason} className="text-[11px] leading-snug text-ink-dim">
                  &middot; {reason}{" "}
                  <span className="tabular text-ink-faint">
                    ({formatPercent(meta.modality.confidence, 0)} confidence)
                  </span>
                </li>
              ))}
            </ul>
          </div>

          {meta.ingest_notes.length > 0 && (
            <ul className="mt-3 space-y-1">
              {meta.ingest_notes.map((note) => (
                <li
                  key={note}
                  className="rounded-md border border-uncertain/30 bg-uncertain/5 px-2.5 py-1.5 text-[11px] leading-snug text-uncertain"
                >
                  {note}
                </li>
              ))}
            </ul>
          )}

          {tagEntries.length > 0 && (
            <div className="mt-3">
              <button
                type="button"
                onClick={() => setShowTags((v) => !v)}
                aria-expanded={showTags}
                className="text-[10px] uppercase tracking-wider text-ink-faint transition hover:text-signal"
              >
                {showTags ? "Hide" : "Show"} file tags ({tagEntries.length})
              </button>
              {showTags && (
                <dl className="mt-1.5 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5">
                  {tagEntries.map(([key, value]) => (
                    <div key={key} className="col-span-2 grid grid-cols-subgrid">
                      <dt className="tabular text-[10px] text-ink-faint">{key}</dt>
                      <dd className="tabular truncate text-[10px] text-ink-dim">
                        {value}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.section>
  );
}
