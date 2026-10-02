import { render, screen } from "@testing-library/react";
import PredictionsSection from "./PredictionsSection";

test("qualifying not done: shows the message, no ranked list", () => {
  render(
    <PredictionsSection
      upcoming={{
        status: "qualifying_not_done",
        next_race: "Bahrain Grand Prix",
        message: "Qualifying for Bahrain Grand Prix hasn't happened yet. Check back after quali.",
      }}
      onLoadFeatures={() => {}}
      featureCache={{}}
    />
  );

  expect(screen.getByText(/hasn't happened yet/i)).toBeInTheDocument();
  expect(screen.queryByText(/%/)).not.toBeInTheDocument();
});

test("qualifying done: renders the ranked win-probability list and winner callout", () => {
  render(
    <PredictionsSection
      upcoming={{
        status: "ok",
        next_race: "Bahrain Grand Prix",
        predictions: [
          { driver_id: 1, code: "VER", full_name: "Max Verstappen", team: "Red Bull", predicted_rank: 1, predicted_probability: 0.4 },
          { driver_id: 2, code: "NOR", full_name: "Lando Norris", team: "McLaren", predicted_rank: 2, predicted_probability: 0.2 },
        ],
      }}
      onLoadFeatures={() => {}}
      featureCache={{}}
    />
  );

  expect(screen.getAllByText("Max Verstappen").length).toBeGreaterThan(0);
  expect(screen.getByText("40.0%")).toBeInTheDocument();
  expect(screen.queryByText(/hasn't happened yet/i)).not.toBeInTheDocument();
});
