/**
 * The four scripted demonstrations.
 *
 * Presenter mode exists because the interesting parts of this system are easy to
 * miss and the boring parts are easy to fumble. It loads the imagery and drafts
 * the contract, then stops. Approval stays a deliberate click: the claim that
 * nothing touches a pixel until a human accepts the plan is the point of the
 * contract, and a demo that auto-approved would be quietly disproving it on
 * camera.
 *
 * The beats say what to look at, never what the number will be. A script that
 * asserted "9.65 km²" would be a figure with no measurement behind it, which is
 * the one thing this system is built not to do — and it would go stale the moment
 * the imagery changed.
 */

export interface DemoBeat {
  /** Which part of the run this beat belongs to. */
  at: "imagery" | "gate" | "contract" | "running" | "answer" | "packet";
  /** What to point at. */
  look: string;
  /** Why it matters. */
  why: string;
}

export interface DemoFollowUp {
  sampleKey: string;
  query: string;
  /** Why the second run settles what the first could not. */
  why: string;
}

export interface DemoScript {
  id: string;
  ordinal: number;
  title: string;
  /** One line on what this demonstration is for. */
  premise: string;
  sampleKey: string;
  query: string;
  /** The expected shape of the answer, in words. Never a number. */
  expect: string;
  beats: DemoBeat[];
  followUp?: DemoFollowUp;
}

export const DEMOS: DemoScript[] = [
  {
    id: "grounding",
    ordinal: 1,
    title: "Point at the thing I named",
    premise:
      "A question in words becomes a region on the map, and every figure beside it says how it was computed.",
    sampleKey: "water_body",
    query: "Highlight the water body referred to in the query",
    expect:
      "A drawn region with its area, pixel count and the published index and threshold that selected it.",
    beats: [
      {
        at: "imagery",
        look: "The band list and the acquisition date",
        why: "Read from the file, never inferred from the filename.",
      },
      {
        at: "gate",
        look: "The readiness checks, each with a measured value and a threshold",
        why: "The gate runs before any analysis, so an unusable input is refused rather than measured badly.",
      },
      {
        at: "contract",
        look: "The index the plan names, and that it says which tools will run",
        why: "The language model plans and phrases. It does not measure.",
      },
      {
        at: "answer",
        look: "Expand the area figure to see its formula and the tool version",
        why: "Every number traces to a computation, and the map layer and the quoted area agree to six decimals.",
      },
    ],
  },
  {
    id: "claim",
    ordinal: 2,
    title: "Test a claim, and agree when it holds",
    premise:
      "A directional claim about change, investigated rather than answered: the system looks for reasons to disbelieve its own finding first, and supports the claim only once they are gone.",
    sampleKey: "reservoir_change",
    query: "Has the water spread increased since the earlier date?",
    expect:
      "A supported verdict with its confidence broken into named components, and the alternative explanations that were tested and ruled out.",
    beats: [
      {
        at: "gate",
        look: "The seasonal offset the gate carries forward",
        why: "Both dates are from the same week of the year before the monsoon, so a difference in stored water is not the season.",
      },
      {
        at: "running",
        look: "Gain, loss and net reported separately",
        why: "A net figure alone hides how much moved in each direction.",
      },
      {
        at: "answer",
        look: "The confidence breakdown, component by component",
        why: "A single percentage cannot be argued with. Each part states what it measured and what would move it.",
      },
      {
        at: "answer",
        look: "Measured twice: the second index that corroborates the first",
        why: "Two independent estimates are compared, not averaged. An average of two disagreeing estimates is a number neither method supports.",
      },
      {
        at: "packet",
        look: "Download the report and check the traceability line",
        why: "Every figure printed in the export is attributed to a measurement or a declared threshold.",
      },
    ],
  },
  {
    id: "debate",
    ordinal: 3,
    title: "Let two instruments disagree",
    premise:
      "Optical and radar of the same ground on the same day. Where they differ is the interesting part, so it is drawn rather than averaged away.",
    sampleKey: "flood_urban",
    query:
      "Use the optical and SAR images together to identify built-up and flooded areas",
    expect:
      "An agreement figure alongside a disagreement map: where both sensors agree, where only radar saw water, and where only optical did.",
    beats: [
      {
        at: "gate",
        look: "The cloud fraction measured from the scene's own classification band",
        why: "The tile metadata claims almost no cloud. The gate measures the area of interest instead, and gets a very different answer.",
      },
      {
        at: "running",
        look: "The radar threshold, and whether Otsu was accepted or refused",
        why: "Otsu always returns something. Outside the physically plausible range for open water it is rejected for a published value, and the reason is stated.",
      },
      {
        at: "answer",
        look: "Both overlap and Cohen's kappa, never one alone",
        why: "Raw overlap flatters any pair on a mostly-dry scene. Kappa removes the agreement you would get by chance.",
      },
      {
        at: "answer",
        look: "Radar water under optical cloud, in its own layer",
        why: "A sensor that could not see is not scored as disagreeing. It is excluded from the agreement figures and counted separately.",
      },
    ],
  },
  {
    id: "seasonal",
    ordinal: 4,
    title: "Refuse, then settle it",
    premise:
      "The scene where a naive system is most confidently wrong. A large, real, measurable change that means nothing like what it appears to mean.",
    sampleKey: "seasonal_farmland",
    query: "Did vegetation decrease between these two dates?",
    expect:
      "A large measured decrease that agrees with the claim, and an inconclusive verdict anyway, because the growing season explains it.",
    beats: [
      {
        at: "running",
        look: "The size of the measured decrease",
        why: "The measurement is real and it agrees with the claim. This is where a system that stopped at measuring would declare success.",
      },
      {
        at: "answer",
        look: "Seasonality, reported as the likely explanation",
        why: "Most of the scene changed at once. Land cover is not repainted across a whole landscape in a season; a crop calendar is.",
      },
      {
        at: "answer",
        look: "The verdict overriding the measurements",
        why: "The alternative explanation is weighed before the evidence, so a finding that cannot be defended is not asserted.",
      },
      {
        at: "answer",
        look: "The imagery offered to settle it",
        why: "The refusal names the acquisition it needs, and the library has it. One click loads it and asks the same question again.",
      },
    ],
    followUp: {
      sampleKey: "seasonal_farmland_same_season",
      query: "Did vegetation decrease between these two dates?",
      why: "The same fields a year apart at the same point in the growing cycle. Phenology cannot account for a difference, so whatever is left is real, and this time the answer changes because the objection is gone rather than because anything was adjusted.",
    },
  },
  {
    id: "unanswerable",
    ordinal: 5,
    title: "Say so when the instrument cannot answer",
    premise:
      "Real construction, and an index that cannot see it. The measurement comes out large and in the claimed direction, and the system still declines to support it.",
    sampleKey: "urban_growth",
    query: "Has the built-up area increased, decreased, or remained unchanged?",
    expect:
      "A large measured increase, and an inconclusive verdict, because the index does not divide this scene into two populations at all.",
    beats: [
      {
        at: "running",
        look: "The separability score for the built-up index",
        why: "At ten metres this index cannot tell concrete from dry soil. Where it scores nothing at splitting the scene, the areas it reports are a cut through a single population.",
      },
      {
        at: "answer",
        look: "Measurement quality, contributing nothing",
        why: "The direction matches the claim and the effect is large, and it is still not supported. An index that does not separate the scene is trusted only when moving its boundary barely moves the answer, and here it moves it a great deal.",
      },
      {
        at: "answer",
        look: "The overlay, against both dates",
        why: "The regions marked as newly built are vegetated on both dates. This is the failure a threshold-only pipeline reports as a confident finding.",
      },
    ],
  },
];

export const DEMOS_BY_ID = new Map(DEMOS.map((demo) => [demo.id, demo]));

/** The beats that belong to one part of the run, in script order. */
export function beatsAt(demo: DemoScript, at: DemoBeat["at"]): DemoBeat[] {
  return demo.beats.filter((beat) => beat.at === at);
}
