import { useSyncExternalStore } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

function supported(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function";
}

function subscribe(onChange: () => void): () => void {
  if (!supported()) return () => {};
  const list = window.matchMedia(QUERY);
  // Older WebKit only exposes the deprecated listener pair.
  if (typeof list.addEventListener === "function") {
    list.addEventListener("change", onChange);
    return () => list.removeEventListener("change", onChange);
  }
  list.addListener(onChange);
  return () => list.removeListener(onChange);
}

function read(): boolean {
  if (!supported()) return false;
  return window.matchMedia(QUERY).matches;
}

/**
 * Reads the user's reduced-motion preference and tracks changes to it.
 *
 * Deliberately a subscription rather than a matchMedia read inside an effect:
 * a component needs to know on its *first* render whether a timed reveal
 * applies. Anything gated behind an animation finishing is then already in the
 * DOM when motion is off, instead of arriving a commit later, which is both
 * what a screen-reader user wants and what keeps the gate testable.
 */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, read, () => false);
}
