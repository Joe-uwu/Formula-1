import { useEffect, useRef, useState } from "react";
import Button3D from "../components/Button3D";
import { TRACK_D, LIGHT_COUNT, easeOutExpo, applyRaceFrame, placeFinishGate } from "../lib/raceIntro";
import "./Hero.css";

// The frame Hero settles to on mount, matching where the loading screen's
// own boot animation leaves off (see LoadingScreen) — Hero no longer plays
// that animation itself, so this is a fixed resting position, not a target
// an on-load timer counts up to.
const INTRO_END_PROGRESS = 0.96;

// Narrative beats: [fadeInStart, fadeInEnd, fadeOutStart, fadeOutEnd]
const BEAT_TITLE = { out: [0.05, 0.12] };
const BEAT_APPROACH = { in: [0.14, 0.2], out: [0.32, 0.38] };
const BEAT_MODEL = { in: [0.4, 0.46], out: [0.58, 0.64] };
const BEAT_BUTTON = { in: [0.9, 0.97] };

const APPROACH_TEXT =
  "Every prediction comes from a model trained on trailing driver and " +
  "constructor form, qualifying position, and circuit history, exactly " +
  "as it stood before each race.";
const MODEL_TEXT =
  "Grid position, qualifying pace, trailing form, and circuit history, " +
  "tested against the simplest baseline there is: always picking the " +
  "pole-sitter.";

function bandOpacity(p, beat) {
  const { in: fadeIn, out: fadeOut } = beat;
  let v = 1;
  if (fadeIn) v = Math.min(v, Math.max(0, (p - fadeIn[0]) / (fadeIn[1] - fadeIn[0])));
  if (fadeOut) v = Math.min(v, Math.max(0, 1 - (p - fadeOut[0]) / (fadeOut[1] - fadeOut[0])));
  return easeOutExpo(Math.min(1, Math.max(0, v)));
}

function usePrefersReducedMotion() {
  const [reduced] = useState(
    () => typeof window !== "undefined" && !!window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
  return reduced;
}

export default function Hero({ status, onReveal, error }) {
  const reducedMotion = usePrefersReducedMotion();
  const scrollerRef = useRef(null);
  const pathRef = useRef(null);
  const carRef = useRef(null);
  const titleRef = useRef(null);
  const approachRef = useRef(null);
  const modelRef = useRef(null);
  const buttonWrapRef = useRef(null);
  const lightRefs = useRef([]);
  const speedLinesRef = useRef(null);
  const flagRef = useRef(null);
  const speedValRef = useRef(null);
  const gearValRef = useRef(null);
  const pathLength = useRef(0);

  useEffect(() => {
    const path = pathRef.current;
    let len = 0;
    if (path) {
      len = placeFinishGate(path, flagRef.current);
      pathLength.current = len;
      if (!reducedMotion) {
        path.style.strokeDasharray = String(len);
        path.style.strokeDashoffset = String(len);
      }
    }

    if (reducedMotion) {
      // Static, fully-settled scene: no scroll listener, everything visible at rest.
      const len = pathLength.current;
      if (path) path.style.strokeDashoffset = "0";
      if (carRef.current && path && len) {
        const end = path.getPointAtLength(len);
        carRef.current.style.transform = `translate(${end.x}px, ${end.y}px)`;
        carRef.current.style.opacity = "1";
      }
      lightRefs.current.forEach((el) => {
        if (el) el.style.fill = "#1a1a1a";
      });
      if (flagRef.current) flagRef.current.style.opacity = "1";
      return;
    }

    // Pulled out of the scroll handler so the instant settle below can apply
    // the same visuals without waiting for a real scroll event.
    const applyProgress = (progress) => {
      if (titleRef.current) {
        const op = bandOpacity(progress, BEAT_TITLE);
        titleRef.current.style.opacity = String(op);
        titleRef.current.style.transform = `translateY(${(1 - op) * -24}px)`;
      }
      setBeat(approachRef.current, progress, BEAT_APPROACH);
      setBeat(modelRef.current, progress, BEAT_MODEL);

      if (buttonWrapRef.current) {
        const op = bandOpacity(progress, BEAT_BUTTON);
        buttonWrapRef.current.style.opacity = String(op);
        buttonWrapRef.current.style.transform = `translateY(${(1 - op) * 20}px)`;
        const revealed = op > 0.05;
        buttonWrapRef.current.style.pointerEvents = revealed ? "auto" : "none";
        const btn = buttonWrapRef.current.querySelector("button");
        if (btn) btn.tabIndex = revealed ? 0 : -1;
      }

      applyRaceFrame(progress, pathLength.current, {
        path,
        car: carRef.current,
        flag: flagRef.current,
        lights: lightRefs.current,
        speedLines: speedLinesRef.current,
        speedVal: speedValRef.current,
        gearVal: gearValRef.current,
      });
    };

    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        ticking = false;
        const el = scrollerRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const scrollable = rect.height - window.innerHeight;
        const progress = scrollable > 0 ? Math.min(1, Math.max(0, -rect.top / scrollable)) : 0;
        applyProgress(progress);
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });

    // The boot animation already played once, full-screen, in the loading
    // screen before this page ever mounted (see LoadingScreen) — so Hero
    // settles straight to that story's resting frame instead of replaying it.
    applyProgress(INTRO_END_PROGRESS);
    const el = scrollerRef.current;
    if (el) {
      const scrollable = el.offsetHeight - window.innerHeight;
      window.scrollTo(0, Math.max(0, scrollable * INTRO_END_PROGRESS));
    }

    return () => {
      window.removeEventListener("scroll", onScroll);
    };
  }, [reducedMotion]);

  const scene = (
    <svg
      className="hero-track-svg"
      viewBox="0 0 1600 900"
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
    >
      <path className="track-path-ghost" d={TRACK_D} fill="none" />
      <path ref={pathRef} className="track-path" d={TRACK_D} fill="none" />

      <g className="start-lights">
        {Array.from({ length: LIGHT_COUNT }).map((_, i) => (
          <circle
            key={i}
            ref={(node) => (lightRefs.current[i] = node)}
            cx={80}
            cy={634 + i * 22}
            r="8"
            className="start-light"
          />
        ))}
      </g>

      <g ref={flagRef} className="finish-gate">
        {Array.from({ length: 10 }).map((_, row) =>
          Array.from({ length: 4 }).map((_, col) => (
            <rect
              key={`${row}-${col}`}
              x={-20 + col * 10}
              y={-70 + row * 14}
              width="10"
              height="14"
              fill={(row + col) % 2 === 0 ? "#f2f2f2" : "#111"}
            />
          ))
        )}
      </g>

      <g ref={carRef} className="car-marker">
        <rect x="-26" y="-9" width="52" height="18" rx="6" className="car-body" />
        <rect x="16" y="-14" width="6" height="28" rx="1" className="car-wing-rear" />
        <rect x="-32" y="-11" width="8" height="22" rx="1" className="car-wing-front" />
        <circle cx="-15" cy="-12" r="5.5" className="car-wheel" />
        <circle cx="-15" cy="12" r="5.5" className="car-wheel" />
        <circle cx="11" cy="-12" r="5.5" className="car-wheel" />
        <circle cx="11" cy="12" r="5.5" className="car-wheel" />
        <ellipse cx="-3" cy="0" rx="8" ry="5" className="car-cockpit" />
      </g>
    </svg>
  );

  if (reducedMotion) {
    return (
      <section className="hero-static">
        <div className="hero-static-scene">{scene}</div>
        <div className="hero-static-content">
          <h1 className="hero-h1">
            Formula 1,<br />predicted.
          </h1>

          <Beat label="The approach" headline="Nothing from after the fact." text={APPROACH_TEXT} />
          <Beat label="The model" headline="Twelve features. One honest baseline." text={MODEL_TEXT} />

          <div className="hero-cta">
            <Button3D onClick={onReveal} disabled={status === "loading"}>
              {status === "loading" ? <span className="btn3d-spinner" aria-label="Loading" /> : "PREDICT"}
            </Button3D>
            {status === "error" && <p className="muted cta-error">Couldn't load predictions: {error}</p>}
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="hero-scroller" ref={scrollerRef}>
      <div className="hero-sticky">
        <div className="hero-speedlines" ref={speedLinesRef} />
        {scene}
        <div className="hero-scrim" />

        <div className="hero-telemetry" aria-hidden="true">
          <div className="telemetry-row">
            <span className="telemetry-label">Speed</span>
            <span className="telemetry-value"><span ref={speedValRef}>0</span> km/h</span>
          </div>
          <div className="telemetry-row">
            <span className="telemetry-label">Gear</span>
            <span className="telemetry-value" ref={gearValRef}>N</span>
          </div>
        </div>

        <div className="hero-ghost hero-ghost-left" aria-hidden="true">F1</div>
        <div className="hero-ghost hero-ghost-right" aria-hidden="true">2026</div>

        <div className="hero-title" ref={titleRef}>
          <h1 className="hero-h1">
            Formula 1,<br />predicted.
          </h1>
        </div>

        <div className="hero-beat" ref={approachRef}>
          <BeatInner label="The approach" headline="Nothing from after the fact." text={APPROACH_TEXT} />
        </div>

        <div className="hero-beat" ref={modelRef}>
          <BeatInner label="The model" headline="Twelve features. One honest baseline." text={MODEL_TEXT} />
        </div>

        <div className="hero-cta" ref={buttonWrapRef}>
          <Button3D onClick={onReveal} disabled={status === "loading"} tabIndex={-1}>
            {status === "loading" ? <span className="btn3d-spinner" aria-label="Loading" /> : "PREDICT"}
          </Button3D>
          {status === "error" && <p className="muted cta-error">Couldn't load predictions: {error}</p>}
        </div>
      </div>
    </section>
  );
}

function setBeat(node, progress, beat) {
  if (!node) return;
  const op = bandOpacity(progress, beat);
  node.style.opacity = String(op);
  node.style.transform = `translateY(${(1 - op) * 24}px)`;
}

function BeatInner({ label, headline, text }) {
  return (
    <>
      <div className="beat-rule" />
      <div className="beat-label">{label}</div>
      <div className="beat-headline">{headline}</div>
      <p className="beat-support">{text}</p>
    </>
  );
}

function Beat({ label, headline, text }) {
  return (
    <div className="hero-beat hero-beat-static">
      <BeatInner label={label} headline={headline} text={text} />
    </div>
  );
}
