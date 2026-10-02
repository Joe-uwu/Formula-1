import { useRef } from "react";
import { motion, useScroll, useTransform } from "motion/react";
import "./MagicText.css";

/** Ported from the pasted magic-text.tsx: each word fades in from a dim
 * ghost copy to full opacity as its slice of the scroll range passes, via
 * Motion's useScroll/useTransform (the "magic" is literally in the scroll
 * offset math, not a hand-rolled version of it). */
function Word({ children, progress, range }) {
  const opacity = useTransform(progress, range, [0, 1]);
  return (
    <span className="magic-word">
      <span className="magic-word-ghost" aria-hidden="true">{children}</span>
      <motion.span style={{ opacity }}>{children}</motion.span>
    </span>
  );
}

export function MagicText({ text, className = "" }) {
  const container = useRef(null);
  const { scrollYProgress } = useScroll({
    target: container,
    offset: ["start 0.9", "start 0.25"],
  });
  const words = text.split(" ");

  return (
    <p ref={container} className={`magic-text ${className}`}>
      {words.map((word, i) => {
        const start = i / words.length;
        const end = start + 1 / words.length;
        return (
          <Word key={i} progress={scrollYProgress} range={[start, end]}>
            {word}
          </Word>
        );
      })}
    </p>
  );
}

export default MagicText;
