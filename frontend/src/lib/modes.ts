import type { ImageRole, InputConfiguration } from "./types";

export interface SlotSpec {
  role: ImageRole;
  title: string;
  hint: string;
}

export interface ModeSpec {
  id: Exclude<InputConfiguration, "incomplete">;
  label: string;
  hint: string;
  slots: SlotSpec[];
}

/**
 * The three input configurations the problem statement defines. Single-image is
 * the mandatory baseline; the paired configurations are the principal focus.
 */
export const MODES: ModeSpec[] = [
  {
    id: "single",
    label: "Single image",
    hint: "VQA, scene description, text-guided grounding",
    slots: [
      {
        role: "single",
        title: "Optical, multispectral, or SAR",
        hint: "GeoTIFF or TIFF \u00b7 PNG/JPEG for benchmarks",
      },
    ],
  },
  {
    id: "cross_modal_pair",
    label: "Optical + SAR",
    hint: "Joint extraction from co-registered modalities",
    slots: [
      {
        role: "optical",
        title: "Optical / multispectral",
        hint: "Sentinel-2 or Cartosat-2S \u00b7 GeoTIFF",
      },
      { role: "sar", title: "SAR", hint: "Sentinel-1 GRD or RISAT \u00b7 GeoTIFF" },
    ],
  },
  {
    id: "bi_temporal_pair",
    label: "Bi-temporal",
    hint: "Change detection, change description, change VQA",
    slots: [
      { role: "date_a", title: "Earlier acquisition", hint: "Date A \u00b7 GeoTIFF" },
      { role: "date_b", title: "Later acquisition", hint: "Date B \u00b7 GeoTIFF" },
    ],
  },
];

export function modeFor(id: InputConfiguration): ModeSpec {
  return MODES.find((mode) => mode.id === id) ?? MODES[0];
}
