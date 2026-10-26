import "./NextRaceSection.css";

const pct = (p) => `${(p * 100).toFixed(1)}%`;

/** The next race's predicted winner, big, with the chasing pack beside it. */
export default function NextRaceSection({ upcoming }) {
  const eyebrow = `Round ${upcoming.round} · ${upcoming.year}`;

  if (upcoming.status !== "ok") {
    return (
      <div className="nr nr-pending">
        <div className="nr-eyebrow">{eyebrow}</div>
        <h2 className="nr-race">{upcoming.next_race}</h2>
        <p className="nr-meta">{upcoming.message}</p>
      </div>
    );
  }

  const [winner, ...rest] = upcoming.predictions;
  const chasers = rest.slice(0, 5);
  const maxProb = winner.predicted_probability || 1;
  const [first, ...lastParts] = winner.full_name.split(" ");
  const last = lastParts.join(" ");

  return (
    <div className="nr">
      <div className="nr-main">
        <div className="nr-eyebrow">{eyebrow} · {upcoming.next_race}</div>
        <div className="nr-label">Predicted winner</div>
        <h2 className="nr-name" aria-label={winner.full_name}>
          <span className="nr-line"><span className="nr-first">{first}</span></span>
          <span className="nr-line"><span className="nr-last" style={{ "--len": last.length }}>{last}</span></span>
        </h2>
        <div className="nr-meta">
          <span>{winner.team}</span>
          <span className="nr-prob">{pct(winner.predicted_probability)}</span>
          <span className="nr-meta-muted">win probability</span>
        </div>
        <div className="nr-bar"><span style={{ "--w": "100%" }} /></div>
      </div>

      <ol className="nr-pack" aria-label="Next most likely winners">
        {chasers.map((p, i) => (
          <li key={p.driver_id} className="nr-row" style={{ "--i": i }}>
            <span className="nr-rank">P{i + 2}</span>
            <span className="nr-who">
              <span className="nr-who-name">{p.full_name}</span>
              <span className="nr-who-team">{p.team}</span>
            </span>
            <span className="nr-row-prob">{pct(p.predicted_probability)}</span>
            <span className="nr-bar nr-bar-thin">
              <span style={{ "--w": `${(p.predicted_probability / maxProb) * 100}%` }} />
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
