/**
 * Every numeric readout in the UI goes through this module.
 *
 * Centralising it serves the Never-Guess Rule in a practical way: units and
 * precision are consistent everywhere, and a value that was never measured
 * renders as an explicit dash rather than a zero that looks like a measurement.
 */

export const NOT_MEASURED = "\u2014";

export function formatNumber(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return NOT_MEASURED;
  }
  return value.toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatInteger(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return NOT_MEASURED;
  }
  return Math.round(value).toLocaleString("en-IN");
}

/** Ground sample distance. Sub-metre resolutions keep two decimals. */
export function formatGsd(metres: number | null | undefined): string {
  if (metres === null || metres === undefined || !Number.isFinite(metres)) {
    return NOT_MEASURED;
  }
  if (metres < 1) return `${metres.toFixed(2)} m`;
  if (metres < 10) return `${metres.toFixed(1)} m`;
  return `${Math.round(metres)} m`;
}

export function formatArea(km2: number | null | undefined): string {
  if (km2 === null || km2 === undefined || !Number.isFinite(km2)) {
    return NOT_MEASURED;
  }
  if (km2 < 0.01) return `${formatNumber(km2 * 1_000_000, 0)} m\u00b2`;
  if (km2 < 1) return `${formatNumber(km2, 3)} km\u00b2`;
  return `${formatNumber(km2, 2)} km\u00b2`;
}

export function formatPercent(
  fraction: number | null | undefined,
  digits = 1,
): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) {
    return NOT_MEASURED;
  }
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) {
    return NOT_MEASURED;
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 && unit > 0 ? 1 : 0)} ${units[unit]}`;
}

/** Decimal degrees with a hemisphere letter, as used on map readouts. */
export function formatLatLon(
  lonLat: number[] | null | undefined,
): string {
  if (!lonLat || lonLat.length < 2) return NOT_MEASURED;
  const [lon, lat] = lonLat;
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(4)}\u00b0${ns}, ${Math.abs(lon).toFixed(4)}\u00b0${ew}`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return NOT_MEASURED;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return NOT_MEASURED;
  return date.toLocaleDateString("en-IN", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return NOT_MEASURED;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatDimensions(width: number, height: number): string {
  return `${width.toLocaleString("en-IN")} \u00d7 ${height.toLocaleString("en-IN")} px`;
}
