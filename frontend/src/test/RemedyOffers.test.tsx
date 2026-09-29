import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RemedyOffers } from "../components/RemedyOffers";
import type { RemedyOffer } from "../lib/types";

function offer(overrides: Partial<RemedyOffer> = {}): RemedyOffer {
  return {
    confounder: "seasonality",
    confounder_label: "Seasonal phenology rather than land-cover change",
    requirement:
      "an acquisition from the same part of the growing season, a year apart",
    sample_key: "seasonal_farmland_same_season",
    title: "Punjab, one year apart in the same season",
    place: "Irrigated cropland north of Moga, Punjab",
    configuration: "bi_temporal_pair",
    why: "Both scenes sit in the same part of the growing cycle, so phenology cannot account for a difference.",
    corrects_current: true,
    suggested_query: "Did vegetation decrease between these two dates?",
    ...overrides,
  };
}

describe("RemedyOffers", () => {
  it("renders nothing when there is nothing left standing", () => {
    const { container } = render(
      <RemedyOffers remedies={{ offers: [], unmet: [] }} onTake={() => {}} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the run carried no remedy set", () => {
    const { container } = render(
      <RemedyOffers remedies={null} onTake={() => {}} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("names the objection and the requirement it answers", () => {
    render(
      <RemedyOffers
        remedies={{ offers: [offer()], unmet: [] }}
        onTake={() => {}}
      />,
    );

    expect(
      screen.getByText(/Seasonal phenology rather than land-cover change/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/same part of the growing season, a year apart/),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Punjab, one year apart in the same season"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Irrigated cropland north of Moga/),
    ).toBeInTheDocument();
  });

  it("explains why the scene settles it, from the scene's own words", () => {
    render(
      <RemedyOffers
        remedies={{ offers: [offer()], unmet: [] }}
        onTake={() => {}}
      />,
    );
    expect(
      screen.getByText(/phenology cannot account for a difference/),
    ).toBeInTheDocument();
  });

  it("marks the scene built for this objection", () => {
    render(
      <RemedyOffers
        remedies={{ offers: [offer()], unmet: [] }}
        onTake={() => {}}
      />,
    );
    expect(screen.getByText(/built for this objection/)).toBeInTheDocument();
  });

  it("does not mark an incidental match as built for the objection", () => {
    render(
      <RemedyOffers
        remedies={{ offers: [offer({ corrects_current: false })], unmet: [] }}
        onTake={() => {}}
      />,
    );
    expect(screen.queryByText(/built for this objection/)).toBeNull();
  });

  it("shows the question it will ask, so the re-run is not a surprise", () => {
    render(
      <RemedyOffers
        remedies={{ offers: [offer()], unmet: [] }}
        onTake={() => {}}
      />,
    );
    expect(
      screen.getByText(/Did vegetation decrease between these two dates\?/),
    ).toBeInTheDocument();
  });

  it("hands the whole offer back when taken", async () => {
    const onTake = vi.fn();
    const only = offer();
    render(
      <RemedyOffers remedies={{ offers: [only], unmet: [] }} onTake={onTake} />,
    );

    await userEvent.click(
      screen.getByRole("button", { name: /Load it and re-ask/ }),
    );
    expect(onTake).toHaveBeenCalledWith(only);
  });

  it("reports loading against the scene being fetched", () => {
    render(
      <RemedyOffers
        remedies={{ offers: [offer()], unmet: [] }}
        onTake={() => {}}
        busyKey="seasonal_farmland_same_season"
      />,
    );
    const button = screen.getByRole("button", { name: /Loading/ });
    expect(button).toBeDisabled();
  });

  it("does not offer to load while a run is already going", () => {
    render(
      <RemedyOffers
        remedies={{ offers: [offer()], unmet: [] }}
        onTake={() => {}}
        disabled
      />,
    );
    expect(
      screen.getByRole("button", { name: /Load it and re-ask/ }),
    ).toBeDisabled();
  });

  it("lists objections nothing on hand can answer", () => {
    render(
      <RemedyOffers
        remedies={{
          offers: [],
          unmet: ["a cloud-free acquisition within a week of the flood peak"],
        }}
        onTake={() => {}}
      />,
    );

    expect(
      screen.getByText(/Still unanswered by anything on hand/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/within a week of the flood peak/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Nothing in the sample library removes/),
    ).toBeInTheDocument();
  });

  it("shows several offers at once without collapsing them", () => {
    render(
      <RemedyOffers
        remedies={{
          offers: [
            offer(),
            offer({
              confounder: "cloud_shadow",
              confounder_label: "Cloud mistaken for change",
              sample_key: "water_body",
              title: "Ukai reservoir",
              corrects_current: false,
              suggested_query: null,
            }),
          ],
          unmet: [],
        }}
        onTake={() => {}}
      />,
    );

    expect(
      screen.getAllByRole("button", { name: /Load it and re-ask/ }),
    ).toHaveLength(2);
    expect(screen.getByText(/Cloud mistaken for change/)).toBeInTheDocument();
  });
});
