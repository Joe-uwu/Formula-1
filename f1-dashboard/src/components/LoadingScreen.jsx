import { useEffect, useRef, useState } from "react";
import {
  TRACK_D,
  LIGHT_COUNT,
  easeOutExpo,
  applyRaceFrame,
  placeFinishGate,
} from "../lib/raceIntro";
import "./LoadingScreen.css";

const DURATION_MS = 3200;
const HOLD_MS = 450; // let "100%" register before the reveal starts
const FADE_MS = 600; // must match .loading-screen's CSS transition duration

function usePrefersReducedMotion() {
  const [reduced] = useState(
    () =>
      typeof window !== "undefined" &&
      !!window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
  return reduced;
}

/** The full-screen splash shown while the site itself is loading: the same
 * racetrack boot sequence that used to autoplay inside Hero, now run once,
 * up front, with a percentage readout tracking its own progress. Once the
 * car crosses the finish gate it holds a beat, fades, and hands off to
 * `onDone` — the landing page underneath is already mounted and settled by
 * then, so the fade reveals it rather than cutting to it. */
export default function LoadingScreen({ onDone }) {
  const reducedMotion = usePrefersReducedMotion();
  const [finishing, setFinishing] = useState(false);
  const pathRef = useRef(null);
  const carRef = useRef(null);
  const flagRef = useRef(null);
  const lightRefs = useRef([]);
  const speedLinesRef = useRef(null);
  const speedValRef = useRef(null);
  const gearValRef = useRef(null);
  const percentValRef = useRef(null);
  const finishedRef = useRef(false);

  useEffect(() => {
    if (reducedMotion) {
      onDone();
      return;
    }

    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const path = pathRef.current;
    const len = path ? placeFinishGate(path, flagRef.current) : 0;
    if (path) {
      path.style.strokeDasharray = String(len);
      path.style.strokeDashoffset = String(len);
    }

    const apply = (progress) => {
      if (percentValRef.current) {
        percentValRef.current.textContent = String(Math.round(progress * 100));
      }
      applyRaceFrame(progress, len, {
        path,
        car: carRef.current,
        flag: flagRef.current,
        lights: lightRefs.current,
        speedLines: speedLinesRef.current,
        speedVal: speedValRef.current,
        gearVal: gearValRef.current,
      });
    };

    const finish = () => {
      if (finishedRef.current) return; // rAF completion and the safety timer can both fire
      finishedRef.current = true;
      apply(1);
      window.setTimeout(() => {
        setFinishing(true);
        window.setTimeout(onDone, FADE_MS);
      }, HOLD_MS);
    };

    let raf;
    const onSkip = () => finish();
    window.addEventListener("pointerdown", onSkip);
    window.addEventListener("keydown", onSkip);
    window.addEventListener("wheel", onSkip, { passive: true });

    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / DURATION_MS);
      apply(easeOutExpo(t));
      if (t >= 1) {
        finish();
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    // rAF throttles to near-zero on a tab that loads without real OS focus —
    // this guarantees the loader can't get stuck open forever.
    const safety = window.setTimeout(finish, DURATION_MS + 1500);

    return () => {
      document.body.style.overflow = prevOverflow;
      window.clearTimeout(safety);
      window.removeEventListener("pointerdown", onSkip);
      window.removeEventListener("keydown", onSkip);
      window.removeEventListener("wheel", onSkip);
      cancelAnimationFrame(raf);
    };
  }, [reducedMotion, onDone]);

  if (reducedMotion) return null;

  return (
    <div className={`loading-screen ${finishing ? "is-finishing" : ""}`}>
      <div className="loading-speedlines" ref={speedLinesRef} />

      <svg
        className="loading-track-svg"
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

      <div className="loading-telemetry" aria-hidden="true">
        <div className="telemetry-row">
          <span className="telemetry-label">Speed</span>
          <span className="telemetry-value">
            <span ref={speedValRef}>0</span> km/h
          </span>
        </div>
        <div className="telemetry-row">
          <span className="telemetry-label">Gear</span>
          <span className="telemetry-value" ref={gearValRef}>
            N
          </span>
        </div>
      </div>

      <div className="loading-percent" role="status" aria-live="polite">
        <span className="loading-percent-value" ref={percentValRef}>
          0
        </span>
        <span className="loading-percent-sign">%</span>
      </div>
    </div>
  );
}
