import { useEffect, useRef } from "react";
import { gsap } from "../lib/gsapSetup";
import "./LiquidBar.css";

/** A track-record bar that fills like rising liquid as you scroll past it —
 * width is scrubbed to scroll position (not a timed animation), so the
 * "pour" tracks scroll position directly and plays the same in reverse. A
 * small wobbling blob at the leading edge plus rising bubbles sell the
 * liquid read; the value itself is always the real percentage passed in. */
export default function LiquidBar({ value, className = "" }) {
  const trackRef = useRef(null);
  const fillRef = useRef(null);

  useEffect(() => {
    const track = trackRef.current;
    const fill = fillRef.current;
    if (!track || !fill) return;

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      fill.style.width = `${value * 100}%`;
      return;
    }

    fill.style.width = "0%";
    const tween = gsap.to(fill, {
      width: `${value * 100}%`,
      ease: "none",
      scrollTrigger: { trigger: track, start: "top 92%", end: "top 55%", scrub: 0.3 },
    });
    return () => {
      tween.scrollTrigger?.kill();
      tween.kill();
    };
  }, [value]);

  return (
    <span className="liquid-track" ref={trackRef}>
      <span className={`liquid-fill ${className}`} ref={fillRef}>
        <span className="liquid-bubble b1" />
        <span className="liquid-bubble b2" />
        <span className="liquid-bubble b3" />
      </span>
    </span>
  );
}
