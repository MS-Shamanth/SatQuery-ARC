import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PresenterPanel } from "../components/PresenterPanel";
import { DEMOS, DEMOS_BY_ID, beatsAt } from "../lib/demos";

const ALL_CACHED = new Set(DEMOS.map((demo) => demo.sampleKey));

function open() {
  return userEvent.click(screen.getByRole("button", { name: /Presenter/ }));
}

describe("the demo scripts", () => {
  it("covers the scripted demonstrations, numbered in order", () => {
    expect(DEMOS.map((demo) => demo.ordinal)).toEqual([1, 2, 3, 4, 5]);
  });

  it("includes a demo where the system declines to answer", () => {
    // A library of only answerable scenes would misrepresent the system. The
    // refusal has to be demonstrable, not described.
    const declined = DEMOS_BY_ID.get("unanswerable");
    expect(declined?.sampleKey).toBe("urban_growth");
    expect(declined?.expect).toMatch(/inconclusive/i);
  });

  it("bases the claim-investigation demo on a scene whose index suits it", () => {
    const claim = DEMOS_BY_ID.get("claim");
    expect(claim?.sampleKey).toBe("reservoir_change");
    expect(claim?.expect).toMatch(/supported/i);
  });

  it("names a sample scene and a question for every demo", () => {
    for (const demo of DEMOS) {
      expect(demo.sampleKey).toBeTruthy();
      expect(demo.query).toBeTruthy();
      expect(demo.beats.length).toBeGreaterThan(0);
    }
  });

  it("never promises a figure, because a scripted number has no measurement behind it", () => {
    // The scripts describe what to look at. A digit in that prose would be a
    // number on screen that no tool produced, and it would go stale the moment
    // the imagery changed.
    for (const demo of DEMOS) {
      const prose = [
        demo.premise,
        demo.expect,
        ...demo.beats.flatMap((beat) => [beat.look, beat.why]),
        demo.followUp?.why ?? "",
      ].join(" ");
      expect(prose, `${demo.id} quotes a figure`).not.toMatch(/\d/);
    }
  });

  it("gives the seasonal demo a second run that settles the first", () => {
    const seasonal = DEMOS_BY_ID.get("seasonal");
    expect(seasonal?.followUp?.sampleKey).toBe("seasonal_farmland_same_season");
    expect(seasonal?.followUp?.query).toBe(seasonal?.query);
  });

  it("selects the beats belonging to one part of the run", () => {
    const claim = DEMOS_BY_ID.get("claim")!;
    expect(beatsAt(claim, "answer").length).toBeGreaterThan(0);
    expect(beatsAt(claim, "answer").every((beat) => beat.at === "answer")).toBe(
      true,
    );
  });
});

describe("PresenterPanel", () => {
  it("stays collapsed until asked, so it does not crowd the workspace", () => {
    render(
      <PresenterPanel
        active={null}
        phase="idle"
        availableKeys={ALL_CACHED}
        onStart={() => {}}
        onClear={() => {}}
      />,
    );
    expect(screen.queryByText(/Point at the thing I named/)).toBeNull();
    expect(screen.getByText(`${DEMOS.length} demos`)).toBeInTheDocument();
  });

  it("lists every demo with its premise once opened", async () => {
    render(
      <PresenterPanel
        active={null}
        phase="idle"
        availableKeys={ALL_CACHED}
        onStart={() => {}}
        onClear={() => {}}
      />,
    );
    await open();
    for (const demo of DEMOS) {
      expect(screen.getByText(demo.title)).toBeInTheDocument();
      expect(screen.getByText(demo.premise)).toBeInTheDocument();
    }
  });

  it("says it will not approve the contract for you", async () => {
    render(
      <PresenterPanel
        active={null}
        phase="idle"
        availableKeys={ALL_CACHED}
        onStart={() => {}}
        onClear={() => {}}
      />,
    );
    await open();
    expect(
      screen.getByText(/nothing measures anything until you accept/),
    ).toBeInTheDocument();
  });

  it("hands back the whole script when a demo is set up", async () => {
    const onStart = vi.fn();
    render(
      <PresenterPanel
        active={null}
        phase="idle"
        availableKeys={ALL_CACHED}
        onStart={onStart}
        onClear={() => {}}
      />,
    );
    await open();
    await userEvent.click(screen.getAllByRole("button", { name: "Set up" })[0]);
    expect(onStart).toHaveBeenCalledWith(DEMOS[0]);
  });

  it("refuses to offer a demo whose imagery is not cached", async () => {
    render(
      <PresenterPanel
        active={null}
        phase="idle"
        availableKeys={new Set()}
        onStart={() => {}}
        onClear={() => {}}
      />,
    );
    await open();
    for (const button of screen.getAllByRole("button", { name: "Set up" })) {
      expect(button).toBeDisabled();
    }
    expect(screen.getAllByText(/not cached/).length).toBe(DEMOS.length);
  });

  it("reports the phase it has reached", async () => {
    render(
      <PresenterPanel
        active={DEMOS_BY_ID.get("claim")!}
        phase="awaiting_approval"
        availableKeys={ALL_CACHED}
        onStart={() => {}}
        onClear={() => {}}
      />,
    );
    expect(
      screen.getByText(/waiting for you to approve the plan/),
    ).toBeInTheDocument();
  });

  it("shows only the beats for the current step", async () => {
    const claim = DEMOS_BY_ID.get("claim")!;
    render(
      <PresenterPanel
        active={claim}
        phase="running"
        availableKeys={ALL_CACHED}
        onStart={() => {}}
        onClear={() => {}}
      />,
    );
    await open();

    const running = beatsAt(claim, "running")[0];
    expect(screen.getByText(running.look)).toBeInTheDocument();
    const answer = beatsAt(claim, "answer")[0];
    expect(screen.queryByText(answer.look)).toBeNull();
  });

  it("brings up the answer beats once the run has settled", async () => {
    const claim = DEMOS_BY_ID.get("claim")!;
    render(
      <PresenterPanel
        active={claim}
        phase="answered"
        availableKeys={ALL_CACHED}
        onStart={() => {}}
        onClear={() => {}}
      />,
    );
    await open();
    expect(screen.getByText(beatsAt(claim, "answer")[0].look)).toBeInTheDocument();
  });

  it("explains the second run only once the first has answered", async () => {
    const seasonal = DEMOS_BY_ID.get("seasonal")!;
    const { rerender } = render(
      <PresenterPanel
        active={seasonal}
        phase="running"
        availableKeys={ALL_CACHED}
        onStart={() => {}}
        onClear={() => {}}
      />,
    );
    await open();
    expect(screen.queryByText(/The second run/)).toBeNull();

    rerender(
      <PresenterPanel
        active={seasonal}
        phase="answered"
        availableKeys={ALL_CACHED}
        onStart={() => {}}
        onClear={() => {}}
      />,
    );
    expect(screen.getByText(/The second run/)).toBeInTheDocument();
    expect(screen.getByText(seasonal.followUp!.why)).toBeInTheDocument();
  });

  it("can be left without reloading the page", async () => {
    const onClear = vi.fn();
    render(
      <PresenterPanel
        active={DEMOS[0]}
        phase="answered"
        availableKeys={ALL_CACHED}
        onStart={() => {}}
        onClear={onClear}
      />,
    );
    await open();
    await userEvent.click(
      screen.getByRole("button", { name: /leave presenter/ }),
    );
    expect(onClear).toHaveBeenCalled();
  });

  it("offers a restart rather than a fresh set up once a demo is under way", async () => {
    render(
      <PresenterPanel
        active={DEMOS[0]}
        phase="running"
        availableKeys={ALL_CACHED}
        onStart={() => {}}
        onClear={() => {}}
      />,
    );
    await open();
    expect(screen.getByRole("button", { name: "Restart" })).toBeInTheDocument();
  });
});
