import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { usePrefersReducedMotion } from "../lib/useReducedMotion";
import { TOUR_STEPS, type TourController, type TourStep } from "../lib/tour";

/**
 * The guided tour overlay: a spotlight on one part of the page and a card
 * explaining it.
 *
 * Two decisions shape this component.
 *
 * Nothing is blocked. The dimmer and the ring are ``pointer-events: none``, so
 * every button the tour points at stays clickable while it is pointing at it.
 * A modal tour would have to say "click Open the Studio" and then refuse the
 * click, which is worse than no tour.
 *
 * A missing target is a state, not a failure. Half these steps describe panels
 * that only exist after the user has loaded imagery and approved a plan. Those
 * steps say what to do to reveal the panel and keep watching for it, so the
 * spotlight snaps onto it the moment it mounts.
 */

const CARD_WIDTH = 344;
const GAP = 14;
const EDGE = 16;

interface Placement {
  top: number;
  left: number;
  /** Which side of the target the card sits on, for the little pointer. */
  side: "top" | "bottom" | "center";
}

function sameRect(a: DOMRect | null, b: DOMRect | null): boolean {
  if (a === null || b === null) return a === b;
  return (
    Math.abs(a.top - b.top) < 0.5 &&
    Math.abs(a.left - b.left) < 0.5 &&
    Math.abs(a.width - b.width) < 0.5 &&
    Math.abs(a.height - b.height) < 0.5
  );
}

function locate(selectors: readonly string[]): HTMLElement | null {
  for (const selector of selectors) {
    const found = document.querySelector<HTMLElement>(selector);
    if (found) return found;
  }
  return null;
}

/**
 * Tracks the highlighted element's box.
 *
 * Polled rather than observed. The targets here appear, grow and reflow while a
 * run streams in, and the set of things that could move one of them is the whole
 * page; a 200ms sample is both simpler than wiring observers to elements that do
 * not exist yet and enough to look continuous. Equal boxes do not re-render.
 */
function useTargetRect(step: TourStep | null): {
  rect: DOMRect | null;
  present: boolean;
} {
  const [rect, setRect] = useState<DOMRect | null>(null);
  const selectors = step?.selectors;

  useEffect(() => {
    if (!selectors) {
      setRect(null);
      return;
    }

    const measure = () => {
      const target = locate(selectors);
      const next = target ? target.getBoundingClientRect() : null;
      setRect((current) => (sameRect(current, next) ? current : next));
    };

    measure();
    const ticker = window.setInterval(measure, 200);
    // Capture phase: the studio scrolls an inner container, not just the window.
    window.addEventListener("scroll", measure, true);
    window.addEventListener("resize", measure);
    return () => {
      window.clearInterval(ticker);
      window.removeEventListener("scroll", measure, true);
      window.removeEventListener("resize", measure);
    };
  }, [selectors]);

  return { rect, present: rect !== null };
}

export function GuidedTour({ tour }: { tour: TourController }) {
  const { step, position, total } = tour;
  const navigate = useNavigate();
  const location = useLocation();
  const reduceMotion = usePrefersReducedMotion();
  const { rect, present } = useTargetRect(step);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [cardHeight, setCardHeight] = useState(240);

  /*
    Route placement, once per step, and then the user leads.

    The obvious version navigates whenever the pathname stops matching the step.
    It is wrong: the second step tells the user to click "Open the Studio", and
    that version pulled them straight back to the landing page the moment they
    did, undoing the one instruction on screen.

    So each step is placed exactly once. After that, a pathname change is the user
    moving, and the tour follows them by jumping to the first step about wherever
    they went.
  */
  const placedFor = useRef<string | null>(null);
  const { goTo } = tour;
  useEffect(() => {
    if (!step) {
      placedFor.current = null;
      return;
    }
    if (placedFor.current !== step.id) {
      placedFor.current = step.id;
      if (location.pathname !== step.route) navigate(step.route);
      return;
    }
    if (location.pathname !== step.route) {
      const ahead = TOUR_STEPS.findIndex(
        (candidate) => candidate.route === location.pathname,
      );
      if (ahead >= 0) goTo(ahead);
    }
  }, [step, location.pathname, navigate, goTo]);

  useLayoutEffect(() => {
    if (cardRef.current) setCardHeight(cardRef.current.offsetHeight);
  }, [step, present]);

  // Bring the target into view when the step opens, and again if it only shows
  // up later. Keyed on the id and on presence, not on the rect, or every scroll
  // would trigger another scroll.
  useEffect(() => {
    if (!step || !present) return;
    const target = locate(step.selectors);
    target?.scrollIntoView({
      block: "center",
      inline: "nearest",
      behavior: reduceMotion ? "auto" : "smooth",
    });
  }, [step, present, reduceMotion]);

  // Move focus to the card so the step is announced and the keys below work
  // without the user having to click first.
  useEffect(() => {
    if (step) cardRef.current?.focus();
  }, [step]);

  const { next, back, skip } = tour;
  useEffect(() => {
    if (!step) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        skip();
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        next();
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        back();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, next, back, skip]);

  const place = useCallback((): Placement => {
    const viewportW = window.innerWidth || 1280;
    const viewportH = window.innerHeight || 800;
    const centered: Placement = {
      top: Math.max(EDGE, viewportH / 2 - cardHeight / 2),
      left: Math.max(EDGE, viewportW / 2 - CARD_WIDTH / 2),
      side: "center",
    };
    if (!rect) return centered;

    const left = Math.min(
      Math.max(EDGE, rect.left + rect.width / 2 - CARD_WIDTH / 2),
      Math.max(EDGE, viewportW - CARD_WIDTH - EDGE),
    );

    const below = rect.bottom + GAP;
    if (below + cardHeight <= viewportH - EDGE) {
      return { top: below, left, side: "top" };
    }
    const above = rect.top - GAP - cardHeight;
    if (above >= EDGE) return { top: above, left, side: "bottom" };
    return { ...centered, left };
  }, [rect, cardHeight]);

  if (!step) return null;

  const placement = place();
  const last = position === total;

  return (
    <div className="pointer-events-none fixed inset-0 z-[9998]" data-tour-overlay="">
      {/*
        The dim. One box at the target with an oversized spread dims everything
        around it and leaves the target at full brightness, without needing an
        SVG mask. With no target, a plain sheet.
      */}
      {rect ? (
        <div
          aria-hidden="true"
          className="absolute rounded-xl"
          style={{
            top: rect.top - 6,
            left: rect.left - 6,
            width: rect.width + 12,
            height: rect.height + 12,
            boxShadow:
              "0 0 0 9999px rgba(3, 6, 12, 0.76), inset 0 0 0 1px var(--color-signal)",
            transition: reduceMotion ? "none" : "all 220ms cubic-bezier(0.16,1,0.3,1)",
          }}
        />
      ) : (
        <div aria-hidden="true" className="absolute inset-0 bg-void/76" />
      )}

      <div
        ref={cardRef}
        role="dialog"
        aria-label="Guided tour"
        aria-live="polite"
        tabIndex={-1}
        className="panel panel-raised pointer-events-auto absolute w-[344px] border border-signal/40 p-4 shadow-2xl outline-none"
        style={{
          top: placement.top,
          left: placement.left,
          transition: reduceMotion ? "none" : "top 220ms cubic-bezier(0.16,1,0.3,1), left 220ms cubic-bezier(0.16,1,0.3,1)",
        }}
      >
        <div className="flex items-center justify-between">
          <span className="tabular text-[10px] uppercase tracking-wider text-signal">
            Step {position} of {total}
          </span>
          <button
            type="button"
            onClick={skip}
            className="rounded px-1.5 py-0.5 text-[10px] text-ink-faint transition hover:text-ink"
          >
            Skip the tour
          </button>
        </div>

        <h2 className="mt-2 text-sm font-semibold leading-snug text-ink">
          {step.title}
        </h2>
        <p className="mt-1.5 text-[11px] leading-relaxed text-ink-dim">{step.body}</p>

        {present && step.action && (
          <p className="mt-2 border-l-2 border-signal/50 pl-2 text-[11px] leading-relaxed text-signal">
            {step.action}
          </p>
        )}

        {/*
          Not present yet. Say why, rather than pointing the spotlight at nothing
          and leaving the user hunting for a panel that cannot be there.
        */}
        {!present && (
          <p className="mt-2 border-l-2 border-uncertain/50 pl-2 text-[11px] leading-relaxed text-uncertain">
            {step.absentHint ?? "This part of the page is not on screen yet."}
          </p>
        )}

        <div className="mt-3.5 flex items-center justify-between gap-2 border-t border-edge pt-3">
          <div className="flex items-center gap-1" aria-hidden="true">
            {Array.from({ length: total }, (_, dot) => (
              <span
                key={dot}
                className={`h-1 rounded-full transition-all ${
                  dot === position - 1
                    ? "w-3.5 bg-signal"
                    : dot < position - 1
                      ? "w-1 bg-signal/45"
                      : "w-1 bg-edge-hi"
                }`}
              />
            ))}
          </div>
          <div className="flex items-center gap-1.5">
            {position > 1 && (
              <button
                type="button"
                onClick={back}
                className="rounded-lg border border-edge px-2.5 py-1.5 text-[11px] text-ink-dim transition hover:border-edge-hi hover:text-ink"
              >
                Back
              </button>
            )}
            <button
              type="button"
              onClick={next}
              className="rounded-lg bg-signal px-3 py-1.5 text-[11px] font-semibold text-void transition hover:brightness-110"
            >
              {last ? "Done" : "Next"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * The return-visit offer.
 *
 * Shown low on the page and easy to ignore, because on a second visit the tour
 * is an offer rather than an introduction. "No thanks" is remembered for good;
 * the Tour button in the header is the way back, so refusing this costs nothing.
 */
export function TourPrompt({ tour }: { tour: TourController }) {
  const reduceMotion = usePrefersReducedMotion();
  if (!tour.prompting) return null;

  return (
    <div
      className="fixed inset-x-0 bottom-0 z-[9997] flex justify-center px-4 pb-4"
      style={{
        animation: reduceMotion
          ? undefined
          : "tour-rise 420ms cubic-bezier(0.16,1,0.3,1) both",
      }}
    >
      <div
        role="region"
        aria-label="Tour offer"
        className="panel panel-raised flex flex-wrap items-center gap-x-3 gap-y-2 border border-signal/35 px-4 py-2.5"
      >
        <p className="text-[11px] leading-snug text-ink-dim">
          Welcome back. Want the tour again?
        </p>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            // Wrapped, not passed straight through: start() takes an optional
            // route and would otherwise receive the click event as one.
            onClick={() => tour.start()}
            className="rounded-lg bg-signal px-3 py-1.5 text-[11px] font-semibold text-void transition hover:brightness-110"
          >
            Take the tour
          </button>
          <button
            type="button"
            onClick={tour.declinePrompt}
            className="rounded-lg border border-edge px-2.5 py-1.5 text-[11px] text-ink-faint transition hover:border-edge-hi hover:text-ink"
          >
            No thanks
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Always-available way into the tour.
 *
 * The return-visit prompt can be refused permanently, so without this the tour
 * would become unreachable after one click.
 */
export function TourButton({
  onStart,
  className = "",
}: {
  onStart: (fromRoute?: string) => void;
  className?: string;
}) {
  // Starts at the first step about the page it was pressed on, so pressing it in
  // the Studio does not throw the user back to the landing page.
  const { pathname } = useLocation();

  return (
    <button
      type="button"
      onClick={() => onStart(pathname)}
      title="Walk through the interface"
      className={`rounded-md border border-edge px-2.5 py-1 text-[10px] uppercase tracking-wide text-ink-faint transition hover:border-signal/50 hover:text-signal ${className}`}
    >
      Tour
    </button>
  );
}
