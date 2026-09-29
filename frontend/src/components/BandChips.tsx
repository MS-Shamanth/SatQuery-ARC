import { motion } from "framer-motion";
import type { BandInfo, BandRole } from "../lib/types";

/**
 * Bands fan out as chips on ingest, each coloured by its spectral region so the
 * composition of a multispectral file is readable at a glance.
 *
 * The colour is descriptive, not semantic: it says "this is a red band", it does
 * not carry the agree/disagree meanings the rest of the palette uses.
 */
const REGION_STYLE: Record<string, string> = {
  visible_blue: "border-sky-400/40 bg-sky-400/10 text-sky-200",
  visible_green: "border-emerald-400/40 bg-emerald-400/10 text-emerald-200",
  visible_red: "border-red-400/40 bg-red-400/10 text-red-200",
  rededge: "border-orange-400/40 bg-orange-400/10 text-orange-200",
  nir: "border-amber-400/40 bg-amber-400/10 text-amber-200",
  swir: "border-rose-400/40 bg-rose-400/10 text-rose-200",
  sar: "border-narrate/40 bg-narrate/10 text-violet-200",
  mask: "border-edge-hi bg-panel-hi text-ink-dim",
  unknown: "border-edge bg-hull text-ink-faint",
};

const ROLE_REGION: Record<BandRole, keyof typeof REGION_STYLE> = {
  coastal: "visible_blue",
  blue: "visible_blue",
  green: "visible_green",
  red: "visible_red",
  rededge1: "rededge",
  rededge2: "rededge",
  rededge3: "rededge",
  nir: "nir",
  nir08: "nir",
  nir09: "nir",
  cirrus: "swir",
  swir16: "swir",
  swir22: "swir",
  scl: "mask",
  vv: "sar",
  vh: "sar",
  hh: "sar",
  hv: "sar",
  gray: "unknown",
  alpha: "mask",
  unknown: "unknown",
};

const ROLE_LABEL: Partial<Record<BandRole, string>> = {
  swir16: "SWIR 1.6",
  swir22: "SWIR 2.2",
  nir08: "NIR 8A",
  nir09: "NIR 9",
  rededge1: "Red edge 1",
  rededge2: "Red edge 2",
  rededge3: "Red edge 3",
  scl: "SCL",
  vv: "VV",
  vh: "VH",
  hh: "HH",
  hv: "HV",
};

function chipLabel(band: BandInfo): string {
  return ROLE_LABEL[band.role] ?? band.role;
}

export function BandChips({ bands }: { bands: BandInfo[] }) {
  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Resolved band roles">
      {bands.map((band, index) => {
        const region = ROLE_REGION[band.role] ?? "unknown";
        const inferred = band.role_source === "convention";
        return (
          <motion.li
            key={band.index}
            initial={{ opacity: 0, y: 8, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{
              delay: index * 0.045,
              duration: 0.35,
              ease: [0.16, 1, 0.3, 1],
            }}
          >
            <span
              className={`inline-flex items-baseline gap-1.5 rounded-md border px-2 py-1 text-[11px] ${REGION_STYLE[region]}`}
              title={
                `Band ${band.index}: "${band.label}" -> ${band.role}` +
                ` (resolved from ${band.role_source})` +
                (band.wavelength_nm ? `, centre ${band.wavelength_nm} nm` : "")
              }
            >
              <span className="tabular opacity-60">{band.index}</span>
              <span className="font-medium">{chipLabel(band)}</span>
              {band.wavelength_nm && (
                <span className="tabular opacity-70">
                  {Math.round(band.wavelength_nm)}nm
                </span>
              )}
              {/* A positional guess is marked, so an inferred role is never
                  mistaken for one the file actually declared. */}
              {inferred && (
                <span
                  className="opacity-70"
                  title="Role inferred from band count convention, not declared in the file"
                >
                  ?
                </span>
              )}
            </span>
          </motion.li>
        );
      })}
    </ul>
  );
}
