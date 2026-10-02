import { useEffect, useRef } from "react";
import { gsap } from "../lib/gsapSetup";
import { TrophyIllustration, PodiumIllustration, HitBadge, MissBadge } from "../components/RaceIllustrations";
import LiquidBar from "../components/LiquidBar";
import "./HistorySection.css";

const METRICS = [
  { key: "hit_at_1", label: "Hit@1", detail: "top pick actually won", Illustration: TrophyIllustration },
  { key: "hit_at_3", label: "Hit@3", detail: "top pick finished top 3", Illustration: PodiumIllustration },
];

export default function HistorySection({ trackRecord, recentRaces }) {
  const metricsRef = useRef(null);
  const listRef = useRef(null);

  // Each line sits in its own overflow-hidden mask and wipes up into view —
  // the reveal-on-scroll treatment from nickho-motorsports.nl's calendar.
  // Scrubbed to scroll position, same as LiquidBar's fill: it tracks scroll
  // directly rather than playing once, so scrolling back up retreats the
  // rows bottom-to-top instead of leaving them stuck on screen.
  useEffect(() => {
    const lines = listRef.current?.querySelectorAll(".race-line");
    if (!lines?.length) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      gsap.set(lines, { yPercent: 0 });
      return;
    }
    const tween = gsap.fromTo(
      lines,
      { yPercent: 100 },
      {
        yPercent: 0, ease: "none", stagger: 0.08,
        scrollTrigger: { trigger: listRef.current, start: "top 92%", end: "top 40%", scrub: 0.3 },
      }
    );
    return () => { tween.scrollTrigger?.kill(); tween.kill(); };
  }, [recentRaces.length]);

  useEffect(() => {
    const cards = metricsRef.current?.querySelectorAll(".metric-reveal");
    if (!cards?.length) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      gsap.set(cards, { opacity: 1 });
      gsap.set(metricsRef.current.querySelectorAll(".metric-illo, .metric-text"), { opacity: 1, x: 0 });
      return;
    }
    const tweens = [...cards].map((card, i) => {
      const fromX = i % 2 === 0 ? -36 : 36;
      return gsap.fromTo(
        card.querySelectorAll(".metric-illo, .metric-text"),
        { opacity: 0, x: fromX },
        {
          opacity: 1, x: 0, duration: 0.7, ease: "expo.out", stagger: 0.15,
          scrollTrigger: { trigger: card, start: "top 82%" },
        }
      );
    });
    return () => tweens.forEach((t) => { t.scrollTrigger?.kill(); t.kill(); });
  }, []);

  return (
    <div className="track-results-grid">
      <div className="track-record-col">
        <div className="track-legend">
          <span className="legend-item"><span className="legend-swatch model" />{trackRecord.model.model_version}</span>
          <span className="legend-item"><span className="legend-swatch pole" />pole_sitter baseline</span>
        </div>

        <div className="metric-groups" ref={metricsRef}>
          {METRICS.map(({ key, label, detail, Illustration }) => (
            <div className="metric-reveal" key={key}>
              <Illustration className="metric-illo" />
              <div className="metric-text">
                <div className="metric-title">{label}: {detail}</div>
                <MetricBar value={trackRecord.model[key]} className="model" />
                <MetricBar value={trackRecord.pole_sitter[key]} className="pole" />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="race-results-col">
        <p className="race-results-caption">Recent race predictions, from the {recentRaces.length} most recent races.</p>
        <div className="race-lines" ref={listRef}>
          {recentRaces.map((r, i) => (
            <div className="race-line-mask" key={r.race_id}>
              <div className={`race-line ${i % 2 === 0 ? "race-line-a" : "race-line-b"}`}>
                <span className="race-line-icon">
                  {r.hit ? <HitBadge className="race-row-icon" /> : <MissBadge className="race-row-icon" />}
                </span>
                <span className="race-line-name">{r.name}</span>
                <span className="race-line-pick">
                  picked <strong>{r.top_pick.full_name}</strong>
                  {r.actual_winner && r.actual_winner.code !== r.top_pick.code && (
                    <> (won by <strong>{r.actual_winner.full_name}</strong>)</>
                  )}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MetricBar({ value, className }) {
  return (
    <div className="metric-bar-row">
      <LiquidBar value={value} className={className} />
      <span className="bar-value">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}
