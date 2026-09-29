import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

/*
  Environment shims below are plain functions rather than vi.fn() on purpose.
  vi.restoreAllMocks() resets mock implementations, which would strip these
  between tests and break any component that touches them on mount.
*/

// jsdom ships no canvas implementation. The starfield is decorative, so a inert
// context keeps component tests focused on behaviour.
const stubContext = {
  clearRect() {},
  fillRect() {},
  beginPath() {},
  arc() {},
  fill() {},
  setTransform() {},
  scale() {},
  createRadialGradient: () => ({ addColorStop() {} }),
} as unknown as CanvasRenderingContext2D;

HTMLCanvasElement.prototype.getContext = (() =>
  stubContext) as unknown as typeof HTMLCanvasElement.prototype.getContext;

// jsdom implements no layout, so scrollIntoView does not exist. The live run log
// pins itself to the newest line with it.
Element.prototype.scrollIntoView = function scrollIntoView() {};

// jsdom has no matchMedia. Reduced motion is reported as preferred so that
// time-based reveals resolve immediately: the typewriter in the contract card
// otherwise adds most of a second to every test that reads past it, and the
// reduced-motion branch is the one that must stay correct anyway.
//
// Exported, and reinstalled after every test, because a test that wants the
// animated path has to replace this and must not leave the replacement behind.
// It did once: a later test inherited motion, waited on a reveal gate it did not
// know about, and failed for a reason that had nothing to do with what it tested.
export function installMatchMedia(prefersReducedMotion = true): void {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    configurable: true,
    value: (query: string): MediaQueryList =>
      ({
        matches: query.includes("prefers-reduced-motion")
          ? prefersReducedMotion
          : false,
        media: query,
        onchange: null,
        addListener() {},
        removeListener() {},
        addEventListener() {},
        removeEventListener() {},
        dispatchEvent: () => false,
      }) as unknown as MediaQueryList,
  });
}

installMatchMedia();

// jsdom's requestAnimationFrame is replaced with a timer-based shim so the
// canvas draw path executes at least once. Handles are tracked and cancelled on
// teardown; otherwise animation frames scheduled by framer-motion fire after the
// component tree is gone and surface as unhandled errors.
const pendingFrames = new Set<number>();

// Frames are tagged with the generation they were scheduled in. Unmounting a
// component can schedule a fresh frame from inside cleanup, which would then run
// against a torn-down tree; dropping stale generations stops that surfacing as
// an unhandled error.
let generation = 0;

window.requestAnimationFrame = ((callback: FrameRequestCallback) => {
  const scheduledIn = generation;
  const handle = window.setTimeout(() => {
    pendingFrames.delete(handle);
    if (scheduledIn !== generation) return;
    callback(performance.now());
  }, 16);
  pendingFrames.add(handle);
  return handle;
}) as typeof requestAnimationFrame;

window.cancelAnimationFrame = ((handle: number) => {
  pendingFrames.delete(handle);
  window.clearTimeout(handle);
}) as typeof cancelAnimationFrame;

afterEach(() => {
  cleanup();
  generation += 1;
  for (const handle of pendingFrames) window.clearTimeout(handle);
  pendingFrames.clear();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  // Undo any per-test motion preference, so the next test starts from the
  // default rather than from whatever the last one needed.
  installMatchMedia();
  localStorage.clear();
});

// jsdom ships no EventSource, and the run stream is the one path in the app that
// depends on it. Without this shim the approve-and-run flow could not be tested
// at all, which is exactly how it came to be the flow that broke: a component
// fault there took the whole page blank and nothing caught it.
//
// Instances register themselves so a test can push events and close the stream
// the way the server does.
class FakeEventSource {
  static instances: FakeEventSource[] = [];

  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onopen: ((event: Event) => void) | null = null;
  readyState = 0;

  constructor(public url: string) {
    FakeEventSource.instances.push(this);
    this.readyState = 1;
  }

  /** Deliver one server-sent event, as the browser would. */
  emit(payload: unknown): void {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }

  /** The server closing the stream, which the browser reports as an error. */
  fail(): void {
    this.onerror?.(new Event("error"));
  }

  close(): void {
    this.readyState = 2;
  }
}

Object.defineProperty(window, "EventSource", {
  writable: true,
  configurable: true,
  value: FakeEventSource,
});

export { FakeEventSource };
