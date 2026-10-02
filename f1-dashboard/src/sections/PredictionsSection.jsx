import { useRef, useState } from "react";
import { useStaggerReveal } from "../lib/useStaggerReveal";
import "./PredictionsSection.css";

const FEATURE_LABELS = {
  grid_position: "Grid position",
  quali_position: "Qualifying position",
  avg_quali_last5: "Avg. qualifying, last 5 races",
  avg_finish_last5: "Avg. finish, last 5 races",
  wins_last5: "Wins, last 5 races",
  avg_finish_last3: "Avg. finish, last 3 races",
  wins_last3: "Wins, last 3 races",
  driver_points_cum: "Career points to date",
  constructor_points_cum: "Constructor points to date",
  constructor_wins_cum: "Constructor wins to date",
  constructor_avg_finish_last3: "Constructor avg. finish, last 3 races",
  circuit_win_rate: "Win rate at this circuit",
};

export default function PredictionsSection({ upcoming, onLoadFeatures, featureCache }) {
  const [expanded, setExpanded] = useState(null);
  const [pendingId, setPendingId] = useState(null);
  const listRef = useRef(null);
  useStaggerReveal(listRef, ".bar-row-wrap", [upcoming.status]);

  if (upcoming.status === "qualifying_not_done") {
    return (
      <div className="card">
        <h3>{upcoming.next_race}</h3>
        <p className="secondary">{upcoming.message}</p>
      </div>
    );
  }

  const predictions = upcoming.predictions;
  const winner = predictions[0];
  const maxProb = winner?.predicted_probability || 1;

  const toggle = async (driverId) => {
    if (expanded === driverId) {
      setExpanded(null);
      return;
    }
    setExpanded(driverId);
    if (!featureCache[driverId]) {
      setPendingId(driverId);
      await onLoadFeatures(driverId);
      setPendingId(null);
    }
  };

  return (
    <div>
      <div className="winner-callout">
        <span className="winner-label">Model's pick to win {upcoming.next_race}</span>
        <span className="winner-name">{winner.full_name}</span>
        <span className="winner-detail muted">{winner.team} · {(winner.predicted_probability * 100).toFixed(1)}% win probability</span>
      </div>

      <div className="bar-list" ref={listRef}>
        {predictions.map((p) => (
          <div key={p.driver_id} className="bar-row-wrap">
            <button className="bar-row" onClick={() => toggle(p.driver_id)}>
              <span className="bar-row-label">
                <span className="bar-rank">{p.predicted_rank}</span>
                <span className="bar-name">{p.full_name}</span>
                <span className="bar-team muted">{p.team}</span>
              </span>
              <span className="bar-track">
                <span className="bar-fill" style={{ width: `${(p.predicted_probability / maxProb) * 100}%` }} />
              </span>
              <span className="bar-value">{(p.predicted_probability * 100).toFixed(1)}%</span>
            </button>

            {expanded === p.driver_id && (
              <div className="feature-panel">
                {pendingId === p.driver_id && <div className="spinner" />}
                {featureCache[p.driver_id]?.error && (
                  <p className="muted">Couldn't load features: {featureCache[p.driver_id].error}</p>
                )}
                {featureCache[p.driver_id]?.features && (
                  <table className="feature-table">
                    <tbody>
                      {Object.entries(featureCache[p.driver_id].features).map(([k, v]) => (
                        <tr key={k}>
                          <td className="muted">{FEATURE_LABELS[k] || k}</td>
                          <td>{v === null ? "N/A" : Number(v).toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
