import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

/**
 * The guided tour: what it points at, and whether it has been seen.
 *
 * The step list lives here rather than in the component so it can be asserted
 * against the real DOM in a test. A step whose target never appears is a broken
 * step, and the only way to know is to look one up by the same selector the
 * component uses.
 *
 * Targets are addressed by ``aria-label`` wherever the panel already carries one.
 * That is a deliberate reuse: those labels are asserted by the panel tests, so
 * they cannot be renamed quietly, and a panel that loses its label has an
 * accessibility problem worth failing over anyway. Where no label exists
 * (the landing hero, the Studio link) a ``data-tour`` hook is added instead.
 */

export type TourRoute = "/" | "/studio";

export interface TourStep {
  /** Stable id, used as a React key and in tests. */
  id: string;
  /** Which page the step belongs to. The tour navigates when this changes. */
  route: TourRoute;
  /** Short heading. */
  title: string;
  /** What the thing is. */
  body: string;
  /** What to click, when there is something to click. */
  action?: string;
  /**
   * Selectors tried in order. The first one present wins, which lets a step
   * point at a panel that renders differently while it loads.
   */
  selectors: string[];
  /**
   * Shown instead of the highlight when no selector matches: this step's target
   * only exists after the user has done something. Without it the tour would
   * silently point at nothing on a page the user has not driven yet.
   */
  absentHint?: string;
}

export const TOUR_STEPS: readonly TourStep[] = [
  {
    id: "what-it-is",
    route: "/",
    title: "SatQuery investigates claims",
    body: "Ask about satellite imagery in plain language. Instead of returning a sentence, SatQuery states the claim it is testing, measures it, tries to explain the result away, and reports what it could and could not settle.",
    action: "Three outcomes are possible, and a forced yes/no is not one of them.",
    selectors: ['[data-tour="landing-hero"]'],
  },
  {
    id: "open-studio",
    route: "/",
    title: "The Studio is where the work happens",
    body: "Everything past this point runs against imagery you load, in a session of your own. Signing in is one click and creates a demo identity in this browser; it guards nothing, and you can skip it.",
    action: "Use this button to go through. Next will take you there too.",
    selectors: ['[data-tour="open-studio"]'],
  },
  {
    id: "presenter",
    route: "/studio",
    title: "Presenter mode is the shortest path",
    body: "Each demo loads a real scene and drafts its question for you. It stops before approval, because a demo that approved its own plan would be quietly skipping the step that matters.",
    action: "Open it and pick a demo if you would rather not load imagery by hand.",
    selectors: ['[aria-label="Presenter mode"]'],
  },
  {
    id: "samples",
    route: "/studio",
    title: "Benchmark scenes, ready to load",
    body: "Sentinel-1 and Sentinel-2 scenes with known content: a reservoir across two years, a flooded city under cloud, farmland across seasons. Each one carries questions worth asking of it.",
    action: "Click any scene to load it into the session.",
    selectors: ['[aria-label="Sample scenes"]'],
  },
  {
    id: "configuration",
    route: "/studio",
    title: "Tell it what kind of input this is",
    body: "One image, two dates of the same place, or an optical and radar pair. The configuration decides which specialists can run at all, so it is chosen before anything is measured.",
    action: "Or drop your own GeoTIFF into a slot below.",
    selectors: ['[aria-label="Input configuration"]'],
  },
  {
    id: "readiness",
    route: "/studio",
    title: "The data is checked before the question is",
    body: "Bands, georeferencing, cloud cover, and whether the two dates actually overlap. This gate can refuse the imagery outright, which is a better answer than a confident measurement of an unusable file.",
    selectors: ['[aria-label="Data readiness"]'],
    absentHint: "Load a scene and this appears, reading the file itself.",
  },
  {
    id: "ask",
    route: "/studio",
    title: "Ask in your own words",
    body: "No syntax to learn. The question is read for what is being claimed and about which target, then turned into a plan you get to see before it runs.",
    action: "Type a question, or click one of the suggestions for the loaded scene.",
    selectors: ['[aria-label="Ask a question"]'],
    absentHint: "Load a scene first, and the question box appears here.",
  },
  {
    id: "contract",
    route: "/studio",
    title: "Approve the plan before it runs",
    body: "The analysis contract names the claim being tested, the tools that will run, the numbers they will produce, and the alternative explanations that will be checked. Nothing executes until you accept it.",
    action: "This is the Approve and run button. It is the one click that starts the pipeline.",
    selectors: ['[aria-label="Analysis contract"]'],
    absentHint: "Ask a question and the contract is drafted here for your approval.",
  },
  {
    id: "disagreement",
    route: "/studio",
    title: "Where the evidence does not agree",
    body: "Green is where every method that could see the area agreed. Red is where they contradicted each other. The red areas are not errors: they are where SatQuery knows the evidence does not agree, and says so instead of picking a side.",
    action: "Click any red region to see what each method concluded, and why they split.",
    selectors: ['[aria-label="Evidence disagreement"]'],
    absentHint: "Approve a contract and this panel reports how much of the result is contested.",
  },
  {
    id: "packet",
    route: "/studio",
    title: "Take the whole investigation with you",
    body: "A PDF report, the execution trace, every measurement with its formula, and the mask layers, each with a checksum. It also lists what it leaves out.",
    action: "Download the archive and every number on screen can be re-checked offline.",
    selectors: ['[aria-label="Evidence packet"]'],
    absentHint: "Finish a run and the packet is assembled here.",
  },
] as const;

export const TOUR_LENGTH = TOUR_STEPS.length;

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

/*
  Versioned keys. If the steps change materially the version moves, and everyone
  is offered the new tour once rather than never seeing it because they finished
  a different one.
*/
const SEEN_KEY = "satquery.tour.v1.seen";
const PROMPT_KEY = "satquery.tour.v1.prompt-declined";

/**
 * localStorage, but never fatal.
 *
 * Access throws outright in a Safari private window and in some embedded
 * webviews. A tour that cannot remember it was shown is a small annoyance; a
 * tour that takes the page down on load is not acceptable.
 */
function readFlag(key: string): boolean {
  try {
    return window.localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}

function writeFlag(key: string, value: boolean): void {
  try {
    if (value) window.localStorage.setItem(key, "1");
    else window.localStorage.removeItem(key);
  } catch {
    /* storage unavailable; the tour just forgets */
  }
}

export interface TourController {
  /** Zero-based index of the current step, or null when the tour is closed. */
  index: number | null;
  step: TourStep | null;
  /** 1-based, for "3 of 10". */
  position: number;
  total: number;
  /** Whether the return-visit prompt should be offered. */
  prompting: boolean;
  /** Open the tour. Given a route, it starts at the first step about that page. */
  start: (fromRoute?: string) => void;
  next: () => void;
  back: () => void;
  /** Leave the tour and remember that it was offered. */
  skip: () => void;
  goTo: (index: number) => void;
  /** Turn down the return-visit prompt for good. */
  declinePrompt: () => void;
}

/**
 * Drives the tour and decides whether to open it unprompted.
 *
 * First visit starts at step one. A later visit does not: it offers the prompt
 * instead, and one refusal is final. Re-opening a tour someone has already
 * dismissed is the fastest way to make the whole thing feel like an obstacle.
 */
export function useTour(): TourController {
  const [index, setIndex] = useState<number | null>(null);
  const [seen, setSeen] = useState(() => readFlag(SEEN_KEY));
  const [declined, setDeclined] = useState(() => readFlag(PROMPT_KEY));

  // Opening happens in an effect, not in the initial state, so the page paints
  // before anything is overlaid on it. A tour that arrives on top of a blank
  // frame is pointing at parts of a page the user has not seen yet.
  useEffect(() => {
    if (seen) return;
    const timer = window.setTimeout(() => setIndex(0), 650);
    return () => window.clearTimeout(timer);
    // Intentionally keyed on nothing: this is the first-visit open, once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const finish = useCallback(() => {
    setIndex(null);
    setSeen(true);
    writeFlag(SEEN_KEY, true);
  }, []);

  const start = useCallback((fromRoute?: string) => {
    // Begin at the first step about the page the user is already on. Pressing
    // "Tour" in the Studio and being teleported to the landing page is a worse
    // introduction than none, and the steps before it are about a page they have
    // evidently already got past.
    const entry = fromRoute
      ? TOUR_STEPS.findIndex((step) => step.route === fromRoute)
      : 0;
    setIndex(entry >= 0 ? entry : 0);
    // Starting by hand counts as having been offered it, and clears the prompt.
    setSeen(true);
    writeFlag(SEEN_KEY, true);
  }, []);

  const next = useCallback(() => {
    setIndex((current) => {
      if (current === null) return null;
      if (current + 1 >= TOUR_LENGTH) {
        setSeen(true);
        writeFlag(SEEN_KEY, true);
        return null;
      }
      return current + 1;
    });
  }, []);

  const back = useCallback(() => {
    setIndex((current) => (current === null ? null : Math.max(0, current - 1)));
  }, []);

  const goTo = useCallback((target: number) => {
    setIndex(Math.min(Math.max(0, target), TOUR_LENGTH - 1));
  }, []);

  const declinePrompt = useCallback(() => {
    setDeclined(true);
    writeFlag(PROMPT_KEY, true);
  }, []);

  const step = index === null ? null : (TOUR_STEPS[index] ?? null);

  return useMemo(
    () => ({
      index,
      step,
      position: index === null ? 0 : index + 1,
      total: TOUR_LENGTH,
      prompting: seen && !declined && index === null,
      start,
      next,
      back,
      skip: finish,
      goTo,
      declinePrompt,
    }),
    [index, step, seen, declined, start, next, back, finish, goTo, declinePrompt],
  );
}

/** Test seam: forget that the tour was ever shown. */
export function resetTourMemory(): void {
  writeFlag(SEEN_KEY, false);
  writeFlag(PROMPT_KEY, false);
}

// ---------------------------------------------------------------------------
// Sharing one controller across the routes
// ---------------------------------------------------------------------------

/*
  The tour crosses from the landing page into the Studio, so its state has to
  outlive either page. It is held above the router and read from here.

  The default is inert rather than null. Panel tests render a page on its own,
  without a provider, and a context that threw there would make the tour a
  precondition for testing everything else.
*/
const INERT: TourController = {
  index: null,
  step: null,
  position: 0,
  total: TOUR_LENGTH,
  prompting: false,
  start: () => {},
  next: () => {},
  back: () => {},
  skip: () => {},
  goTo: () => {},
  declinePrompt: () => {},
};

export const TourContext = createContext<TourController>(INERT);

export function useTourControls(): TourController {
  return useContext(TourContext);
}
