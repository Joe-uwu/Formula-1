import { useEffect } from "react";
import { gsap } from "./gsapSetup";

/** Reveals `selector`'s matches inside `containerRef` with a staggered
 * rise-and-fade the first time the container scrolls into view. Re-runs when
 * `deps` changes (e.g. once real rows replace a loading state). No-op under
 * reduced motion — rows are simply visible immediately. */
export function useStaggerReveal(containerRef, selector, deps = []) {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const items = container.querySelectorAll(selector);
    if (!items.length) return;

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      gsap.set(items, { opacity: 1, y: 0 });
      return;
    }

    const tween = gsap.fromTo(
      items,
      { opacity: 0, y: 18 },
      {
        opacity: 1,
        y: 0,
        duration: 0.5,
        ease: "expo.out",
        stagger: 0.04,
        scrollTrigger: { trigger: container, start: "top 85%" },
      }
    );
    return () => {
      tween.scrollTrigger?.kill();
      tween.kill();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
