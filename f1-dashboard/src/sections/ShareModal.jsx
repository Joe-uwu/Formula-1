import { useEffect, useState } from "react";
import { api } from "../api";
import "./ShareModal.css";

export default function ShareModal({ raceId, onClose }) {
  const [state, setState] = useState({ loading: true, error: null, data: null });

  useEffect(() => {
    let mounted = true;
    api.shareCard(raceId)
      .then((data) => mounted && setState({ loading: false, error: null, data }))
      .catch((e) => mounted && setState({ loading: false, error: e.message, data: null }));
    return () => { mounted = false; };
  }, [raceId]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="share-modal-backdrop" onClick={onClose}>
      <div className="share-modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="share-modal-close" onClick={onClose} aria-label="Close">×</button>

        {state.loading && <div className="spinner" />}
        {state.error && <p className="muted">{state.error}</p>}

        {state.data && (() => {
          const { race, top_pick: topPick, actual_winner: actualWinner } = state.data;
          const hit = actualWinner && actualWinner.code === topPick.code;
          return (
            <>
              <p className="muted share-modal-hint">Screenshot this card to share it.</p>
              <div className="share-card">
                <div className="share-card-header">
                  <span className="share-card-brand">F1 Predictor</span>
                  <span className="share-card-race">{race.name} ({race.date})</span>
                </div>

                <div className="share-card-pick">
                  <div className="share-card-pick-label">Model's pick to win</div>
                  <div className="share-card-pick-name">{topPick.full_name}</div>
                  <div className="share-card-pick-team muted">{topPick.team}</div>
                  <div className="share-card-pick-prob">{(topPick.predicted_probability * 100).toFixed(1)}% win probability</div>
                </div>

                <div className={`share-card-result ${hit ? "hit" : "miss"}`}>
                  {actualWinner ? (
                    hit
                      ? `Correct: ${actualWinner.full_name} won.`
                      : `Actual winner: ${actualWinner.full_name} (model had them ranked #${actualWinner.predicted_rank}).`
                  ) : "Race not yet run."}
                </div>
              </div>
            </>
          );
        })()}
      </div>
    </div>
  );
}
