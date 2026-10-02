import { useEffect, useRef } from "react";
import Lenis from "lenis";
import { gsap, ScrollTrigger } from "./gsapSetup";

/** Global inertia-smoothed scrolling (Lenis), wired into GSAP's ticker so
 * ScrollTrigger stays in sync — skipped entirely under prefers-reduced-motion,
 * where native (instant) scrolling is the more accessible default. Returns a
 * ref to the Lenis instance (null when disabled) for callers that need to
 * scroll programmatically (e.g. `lenisRef.current?.scrollTo(el)`). */
export function useLenis() {
  const lenisRef = useRef(null);

  useEffect(() => {
    // Confirmed by testing: Lenis doesn't move the real document scroll
    // position at all — window.scrollY changes, but no native 'scroll' event
    // ever fires (a plain window listener added purely to check this saw
    // zero calls across a real scroll). So ScrollTrigger has exactly one path
    // to hear about scroll: lenis.on("scroll", ScrollTrigger.update) below,
    // which only fires while gsap.ticker is actually ticking — and
    // gsap.ticker is requestAnimationFrame-driven, which a tab throttles to
    // near-zero without real OS focus. When that happens every ScrollTrigger
    // instance (the nav's active-section tracking included) freezes at
    // whatever it last saw and never recovers on its own. A cheap interval,
    // not rAF-gated the same way, guarantees it can't stay stuck for more
    // than a fraction of a second regardless of tab focus.
    const pollId = window.setInterval(() => ScrollTrigger.update(), 150);

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return () => window.clearInterval(pollId);
    }

    const lenis = new Lenis({ duration: 1.1, smoothWheel: true });
    lenisRef.current = lenis;

    const onTick = (time) => lenis.raf(time * 1000);
    gsap.ticker.add(onTick);
    gsap.ticker.lagSmoothing(0);
    lenis.on("scroll", ScrollTrigger.update);

    return () => {
      window.clearInterval(pollId);
      gsap.ticker.remove(onTick);
      lenis.destroy();
      lenisRef.current = null;
    };
  }, []);

  return lenisRef;
}
