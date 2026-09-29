/**
 * The demo identity and the account menu.
 *
 * The most important assertion in this file is the one that checks the interface
 * says so. This is not authentication: no password, no token, no server call, and
 * nothing is protected. A padlock that guards nothing is only acceptable while it
 * is labelled, so the label is tested like any other behaviour.
 *
 * The rest is the shape the user asked for: one button in, a profile menu in the
 * corner with profile, settings and sign out, and an identity that survives a
 * reload.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { DEMO_USER, initialsOf } from "../lib/auth";

const SESSION_ID = "a".repeat(32);

function mockApi() {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      let body: unknown = {};

      if (url.includes("/readiness")) {
        body = { verdict: "ready", checks: [], notes: [], duration_ms: 10 };
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

/** The tour opens itself on a first visit and would sit over the buttons. */
function tourAlreadySeen() {
  localStorage.setItem("satquery.tour.v1.seen", "1");
  localStorage.setItem("satquery.tour.v1.prompt-declined", "1");
}

const openMenu = async () =>
  userEvent.click(
    await screen.findByRole("button", { name: /Account menu for/i }),
  );

describe("signing in", () => {
  beforeEach(() => {
    mockApi();
    tourAlreadySeen();
  });

  it("offers one button and no form", async () => {
    renderApp();
    expect(
      await screen.findByRole("button", { name: /Sign in to the demo/i }),
    ).toBeInTheDocument();
    // Nothing to type. A demo that asks for a password is a demo nobody finishes.
    expect(screen.queryByLabelText(/password/i)).toBeNull();
    expect(screen.queryByRole("textbox")).toBeNull();
  });

  it("goes straight to the Studio", async () => {
    renderApp();
    await userEvent.click(
      await screen.findByRole("button", { name: /Sign in to the demo/i }),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Input configuration")).toBeInTheDocument(),
    );
  });

  it("lets the Studio be opened without signing in at all", async () => {
    // There is nothing to protect, so the door is not locked.
    renderApp();
    await userEvent.click(
      await screen.findByRole("link", { name: /Skip, just open the Studio/i }),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Input configuration")).toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: "Sign in" })).toBeInTheDocument();
  });

  it("is remembered across a reload", async () => {
    const first = renderApp();
    await userEvent.click(
      await screen.findByRole("button", { name: /Sign in to the demo/i }),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Input configuration")).toBeInTheDocument(),
    );
    first.unmount();

    renderApp("/studio");
    expect(
      await screen.findByRole("button", { name: /Account menu for/i }),
    ).toBeInTheDocument();
  });
});

describe("the account menu", () => {
  beforeEach(() => {
    mockApi();
    tourAlreadySeen();
    localStorage.setItem(
      "satquery.demo.identity.v1",
      JSON.stringify({ ...DEMO_USER, signedInAt: "2026-09-24T09:00:00Z" }),
    );
  });

  it("sits in the Studio header with the user's initials", async () => {
    renderApp("/studio");
    const button = await screen.findByRole("button", { name: /Account menu for/i });
    expect(button).toHaveTextContent(
      initialsOf({ ...DEMO_USER, signedInAt: "" }),
    );
  });

  it("offers profile, settings and sign out", async () => {
    renderApp("/studio");
    await openMenu();

    const menu = screen.getByRole("menu", { name: /account/i });
    expect(within(menu).getByRole("menuitem", { name: /Profile/i })).toBeInTheDocument();
    expect(within(menu).getByRole("menuitem", { name: /Settings/i })).toBeInTheDocument();
    expect(within(menu).getByRole("menuitem", { name: /Sign out/i })).toBeInTheDocument();
  });

  it("says in the interface that it protects nothing", async () => {
    renderApp("/studio");
    await openMenu();
    await userEvent.click(screen.getByRole("menuitem", { name: /Profile/i }));

    expect(screen.getByText(/Demo identity only/i)).toBeInTheDocument();
    expect(screen.getByText(/server never sees it/i)).toBeInTheDocument();
    expect(screen.getByText(/protects nothing/i)).toBeInTheDocument();
  });

  it("offers only settings that do something", async () => {
    renderApp("/studio");
    await openMenu();
    await userEvent.click(screen.getByRole("menuitem", { name: /Settings/i }));

    expect(screen.getByText("Guided tour")).toBeInTheDocument();
    expect(screen.getByText(/Offer the tour again/i)).toBeInTheDocument();
    // And it says where the things it cannot change actually live.
    expect(
      screen.getByText(/held in the backend environment/i),
    ).toBeInTheDocument();
  });

  it("starts the tour from settings", async () => {
    renderApp("/studio");
    await openMenu();
    await userEvent.click(screen.getByRole("menuitem", { name: /Settings/i }));
    await userEvent.click(screen.getByRole("button", { name: "Start" }));

    expect(
      await screen.findByRole("dialog", { name: /guided tour/i }, { timeout: 4000 }),
    ).toBeInTheDocument();
    // The menu gets out of the way of the thing it just opened.
    expect(screen.queryByRole("menu", { name: /account/i })).toBeNull();
  });

  it("signs out, returns to the landing page, and forgets", async () => {
    const first = renderApp("/studio");
    await openMenu();
    await userEvent.click(screen.getByRole("menuitem", { name: /Sign out/i }));

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Sign in to the demo/i }),
      ).toBeInTheDocument(),
    );

    first.unmount();
    renderApp("/studio");
    expect(await screen.findByRole("button", { name: "Sign in" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Account menu for/i })).toBeNull();
  });

  it("closes on Escape", async () => {
    renderApp("/studio");
    await openMenu();
    expect(screen.getByRole("menu", { name: /account/i })).toBeInTheDocument();

    await userEvent.keyboard("{Escape}");
    await waitFor(() =>
      expect(screen.queryByRole("menu", { name: /account/i })).toBeNull(),
    );
  });
});
