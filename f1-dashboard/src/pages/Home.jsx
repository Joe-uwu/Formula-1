import { useCallback, useRef, useState } from "react";
import Hero from "../sections/Hero";
import HistorySection from "../sections/HistorySection";
import CarViewer from "../components/CarViewer";
import MagicText from "../components/MagicText";
import WinnerSignature from "../components/WinnerSignature";
import { api } from "../api";
import { useLenis } from "../lib/smoothScroll";
import "./Home.css";

const TRACK_RECORD_INTRO =
  "Measured against the simplest possible guess using only what was known before lights out";

export default function Home() {
  const [status, setStatus] = useState("idle"); // idle | loading | error | revealed
  const [error, setError] = useState(null);
  const [upcoming, setUpcoming] = useState(null);
  const [trackRecord, setTrackRecord] = useState(null);
  const [recentRaces, setRecentRaces] = useState(null);
  const [lightsOn, setLightsOn] = useState(false);
  const revealRef = useRef(null);
  useLenis();

  const reveal = useCallback(async () => {
    setStatus("loading");
    try {
      const [u, t, r] = await Promise.all([
        api.upcoming(),
        api.trackRecord(10),
        api.recentRaces(8),
      ]);
      setUpcoming(u);
      setTrackRecord(t);
      setRecentRaces(r.races);
      setStatus("revealed");
      // Native scrollIntoView, not lenis.scrollTo: it doesn't depend on the
      // Lenis/gsap ticker still being driven at exactly this moment, so it
      // can't silently stall partway. Two rAFs make sure the just-mounted
      // reveal content (Vanta canvas included) has actually laid out before
      // we measure where to scroll.
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          revealRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        });
      });
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  }, []);

  return (
    <>
      {/* Persistent brand mark — always mounted, always fixed, so it's on
          screen at every scroll position instead of scrolling away with the
          hero's own sticky region once the reveal content starts. */}
      <header className="site-header" aria-hidden="true">Formula 1 Predictor</header>

      <Hero status={status} onReveal={reveal} error={error} />

      {status === "revealed" && (
        <div className="reveal" ref={revealRef}>
          <section className="reveal-section next-race-wrap">
            <div className="next-race-hero">
              <div className="next-race-split">
                <h2 className="next-race-label">
                  <span>Next</span>
                  <span>Race</span>
                  <span>Winner</span>
                </h2>
                <div className="next-race-panel">
                  {upcoming?.status === "qualifying_not_done" ? (
                    <>
                      <div className="next-race-name">{upcoming.next_race}</div>
                      <p className="next-race-message">{upcoming.message}</p>
                    </>
                  ) : upcoming?.status === "ok" ? (
                    <>
                      <div className="next-race-name">{upcoming.next_race}</div>
                      <div className="winner-signature-wrap">
                        <WinnerSignature name={upcoming.predictions[0].full_name} />
                      </div>
                      <p className="next-race-message">
                        {upcoming.predictions[0].team}, {(upcoming.predictions[0].predicted_probability * 100).toFixed(1)}% win probability
                      </p>
                    </>
                  ) : (
                    <div className="spinner" />
                  )}
                </div>
              </div>
            </div>
          </section>

          {/* The intro line gets the whole screen to itself. */}
          <section className="reveal-section track-intro-page">
            <div className="track-intro-block">
              <MagicText text={TRACK_RECORD_INTRO} className="track-intro-text" />
            </div>
          </section>

          {/* Graphs left, predictions right — its own whole page, back to a
              plain two-column layout now that the model has a page of its
              own below. */}
          <section className="page reveal-section track-results-page">
            {trackRecord && recentRaces && (
              <HistorySection trackRecord={trackRecord} recentRaces={recentRaces} />
            )}
          </section>

          {/* The model gets a whole page to itself, last — dim and unlit
              until the bulb is switched on, exactly like it was before it
              had to share a page with the stat columns. */}
          <section className="reveal-section model-page">
            <CarViewer lightsOn={lightsOn} onToggle={() => setLightsOn((v) => !v)} />
          </section>
        </div>
      )}
    </>
  );
}
