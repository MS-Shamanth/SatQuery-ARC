/**
 * Guided tour.
 *
 * The valuable test here is not that a card appears. It is that every step's
 * selector resolves against the page the step claims to be about, because the
 * tour points at panels by ``aria-label`` and a renamed label would turn a step
 * into a spotlight on nothing. The steps that describe panels which only exist
 * after a run are checked the other way: they must say how to reveal them.
 *
 * The rest is the memory rules the user asked for: it opens uninvited exactly
 * once, the offer on a later visit can be refused permanently, and there is
 * always a way back in.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { TOUR_LENGTH, TOUR_STEPS } from "../lib/tour";

const SESSION_ID = "a".repeat(32);

/** Enough of the API for the landing page and an empty studio to render. */
function mockApi() {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      let body: unknown = {};

      if (url.includes("/readiness")) {
        body = { verdict: "ready", checks: [], notes: [], duration_ms: 12 };
      } else if (url.endsWith("/tools")) {
        body = url.includes("/sessions/")
          ? {
              session_id: SESSION_ID,
              configuration: "incomplete",
              available: [],
              unavailable: [],
              declined_registration: {},
            }
          : [];
      } else if (url.includes("/api/samples")) {
        body = { scenes: [], notes: [] };
      } else if (url.includes("/api/health")) {
        body = {
          status: "ok",
          app: "SatQuery AI",
          version: "0.1.0",
          environment: "test",
          components: {},
          capabilities: {
            contract_source: "language-model",
            narration_available: true,
            llm_provider: "openrouter",
            llm_models: ["openrouter/free"],
            sar_source: "copernicus-sentinel-1-grd",
            optical_source: "earth-search-sentinel-2-l2a",
            geospatial_compute: true,
          },
        };
      } else if (url.includes("/api/sessions")) {
        body = {
          session_id: SESSION_ID,
          created_at: "2026-09-21T10:00:00Z",
          updated_at: "2026-09-21T10:00:00Z",
          configuration: "incomplete",
          images: {},
          notes: [],
        };
      }

      return Promise.resolve({
        ok: true,
        status: (init?.method ?? "GET") === "POST" ? 201 : 200,
        statusText: "OK",
        json: () => Promise.resolve(body),
        text: () => Promise.resolve(JSON.stringify(body)),
      } as Response);
    }),
  );
}

const renderApp = (at = "/") =>
  render(
    <MemoryRouter initialEntries={[at]}>
      <App />
    </MemoryRouter>,
  );

/** The tour opens on a timer so the page paints first. */
const findStepCard = () =>
  screen.findByRole("dialog", { name: /guided tour/i }, { timeout: 4000 });

describe("the step list", () => {
  it("is ten steps, as promised in the counter", () => {
    expect(TOUR_LENGTH).toBe(10);
    expect(TOUR_STEPS).toHaveLength(10);
  });

  it("gives every step a unique id and at least one selector", () => {
    const ids = TOUR_STEPS.map((step) => step.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const step of TOUR_STEPS) {
      expect(step.selectors.length).toBeGreaterThan(0);
      expect(step.title).toBeTruthy();
      expect(step.body).toBeTruthy();
    }
  });

  it("explains how to reveal every panel that is not there on arrival", () => {
    // These describe panels that only exist after imagery is loaded and a
    // contract approved. Without a hint the spotlight points at nothing.
    for (const id of ["readiness", "ask", "contract", "disagreement", "packet"]) {
      const step = TOUR_STEPS.find((candidate) => candidate.id === id);
      expect(step, `${id} is missing from the tour`).toBeDefined();
      expect(step?.absentHint, `${id} has no hint for when it is absent`).toBeTruthy();
    }
  });

  it("makes the disagreement panel the step that carries the framing", () => {
    const step = TOUR_STEPS.find((candidate) => candidate.id === "disagreement");
    expect(step?.body).toMatch(/not errors/i);
    expect(step?.body).toMatch(/does not agree/i);
    expect(step?.action).toMatch(/click/i);
  });
});

describe("first visit", () => {
  beforeEach(() => mockApi());

  it("opens on its own and starts at step one of ten", async () => {
    renderApp();
    const card = await findStepCard();
    expect(card).toHaveTextContent("Step 1 of 10");
    expect(card).toHaveTextContent(/SatQuery investigates claims/i);
  });

  it("finds the landing page target it claims to point at", async () => {
    renderApp();
    await findStepCard();
    // The anchor the first step spotlights has to exist, or the dim covers the
    // whole screen and highlights nothing.
    expect(document.querySelector('[data-tour="landing-hero"]')).not.toBeNull();
    expect(screen.queryByText(/not on screen yet/i)).toBeNull();
  });

  it("walks forward and reaches the Studio by the third step", async () => {
    renderApp();
    await findStepCard();

    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(await findStepCard()).toHaveTextContent("Step 2 of 10");
    expect(document.querySelector('[data-tour="open-studio"]')).not.toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    // Step three is about the Studio, so the tour has to have taken us there.
    await waitFor(() =>
      expect(screen.getByLabelText("Input configuration")).toBeInTheDocument(),
    );
    expect(await findStepCard()).toHaveTextContent("Step 3 of 10");
  });

  it("goes back without losing its place", async () => {
    renderApp();
    await findStepCard();
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(await findStepCard()).toHaveTextContent("Step 2 of 10");

    await userEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(await findStepCard()).toHaveTextContent("Step 1 of 10");
    // Nothing to go back to on the first step.
    expect(screen.queryByRole("button", { name: "Back" })).toBeNull();
  });

  it("does not block the button it is pointing at", async () => {
    // A modal tour would say "click Open the Studio" and then swallow the click.
    renderApp();
    await findStepCard();
    const overlay = document.querySelector("[data-tour-overlay]");
    expect(overlay).not.toBeNull();
    // Asserted on the class, not on getComputedStyle: jsdom loads no stylesheet,
    // so every Tailwind utility resolves to the property default there.
    expect(overlay).toHaveClass("pointer-events-none");

    // The behaviour itself, which is what the class is for.
    await userEvent.click(screen.getByRole("link", { name: /Open the Studio/i }));
    await waitFor(() =>
      expect(screen.getByLabelText("Input configuration")).toBeInTheDocument(),
    );
  });

  it("can be skipped, and does not come back uninvited", async () => {
    const first = renderApp();
    await findStepCard();
    await userEvent.click(screen.getByRole("button", { name: /Skip the tour/i }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: /guided tour/i })).toBeNull(),
    );

    first.unmount();
    renderApp();
    // The offer, not the tour.
    expect(await screen.findByRole("region", { name: /tour offer/i })).toBeInTheDocument();
    await new Promise((resolve) => setTimeout(resolve, 900));
    expect(screen.queryByRole("dialog", { name: /guided tour/i })).toBeNull();
  });

  it("closes on Escape", async () => {
    renderApp();
    await findStepCard();
    await userEvent.keyboard("{Escape}");
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: /guided tour/i })).toBeNull(),
    );
  });

  it("ends on the last step rather than wrapping round", async () => {
    renderApp("/studio");
    await findStepCard();
    for (let step = 1; step < TOUR_LENGTH; step += 1) {
      await userEvent.click(screen.getByRole("button", { name: "Next" }));
    }
    expect(await findStepCard()).toHaveTextContent(`Step ${TOUR_LENGTH} of ${TOUR_LENGTH}`);

    await userEvent.click(screen.getByRole("button", { name: "Done" }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: /guided tour/i })).toBeNull(),
    );
  });
});

describe("a later visit", () => {
  beforeEach(() => {
    mockApi();
    // What a returning user's browser looks like.
    localStorage.setItem("satquery.tour.v1.seen", "1");
  });

  it("offers the tour instead of starting it", async () => {
    renderApp();
    const offer = await screen.findByRole("region", { name: /tour offer/i });
    expect(offer).toHaveTextContent(/Want the tour again\?/i);
    expect(screen.queryByRole("dialog", { name: /guided tour/i })).toBeNull();
  });

  it("starts from the offer when accepted", async () => {
    renderApp();
    await screen.findByRole("region", { name: /tour offer/i });
    await userEvent.click(screen.getByRole("button", { name: /Take the tour/i }));

    expect(await findStepCard()).toHaveTextContent("Step 1 of 10");
    expect(screen.queryByRole("region", { name: /tour offer/i })).toBeNull();
  });

  it("stops asking for good once refused", async () => {
    const first = renderApp();
    await screen.findByRole("region", { name: /tour offer/i });
    await userEvent.click(screen.getByRole("button", { name: /No thanks/i }));
    expect(screen.queryByRole("region", { name: /tour offer/i })).toBeNull();

    first.unmount();
    renderApp();
    await screen.findByLabelText("Back to the landing page").catch(() => null);
    await waitFor(() =>
      expect(screen.queryByRole("region", { name: /tour offer/i })).toBeNull(),
    );
  });

  it("keeps a way back in after the offer is refused", async () => {
    // Otherwise "No thanks" makes the tour permanently unreachable.
    localStorage.setItem("satquery.tour.v1.prompt-declined", "1");
    renderApp();

    await userEvent.click(await screen.findByRole("button", { name: "Tour" }));
    expect(await findStepCard()).toHaveTextContent("Step 1 of 10");
  });

  it("offers the same way back from the Studio", async () => {
    localStorage.setItem("satquery.tour.v1.prompt-declined", "1");
    renderApp("/studio");

    await userEvent.click(await screen.findByRole("button", { name: "Tour" }));
    expect(await findStepCard()).toHaveTextContent(/Step \d+ of 10/);
  });
});

describe("a step whose panel is not on screen", () => {
  beforeEach(() => {
    mockApi();
    localStorage.setItem("satquery.tour.v1.seen", "1");
    localStorage.setItem("satquery.tour.v1.prompt-declined", "1");
  });

  it("says what to do instead of pointing at nothing", async () => {
    renderApp("/studio");
    await userEvent.click(await screen.findByRole("button", { name: "Tour" }));
    await findStepCard();

    // Walk to the disagreement step, which needs a finished run to exist. Driven
    // by the heading rather than by a step count, because pressing Tour in the
    // Studio deliberately skips the landing steps.
    const target = TOUR_STEPS.findIndex((step) => step.id === "disagreement");
    for (let click = 0; click < TOUR_LENGTH; click += 1) {
      const current = await findStepCard();
      if (current.textContent?.includes(TOUR_STEPS[target].title)) break;
      await userEvent.click(screen.getByRole("button", { name: "Next" }));
    }

    const card = await findStepCard();
    expect(card).toHaveTextContent(/Where the evidence does not agree/i);
    expect(card).toHaveTextContent(/Approve a contract and this panel reports/i);
    // And the instruction to click a red region is withheld, because there is
    // nothing to click yet.
    expect(card).not.toHaveTextContent(/Click any red region/i);
  });
});
