import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { gsap } from "../lib/gsapSetup";
import "./WinnerSignature.css";

/** The predicted winner's name, written in a script font and revealed left
 * to right like ink from a pen — a rectangular clip-path animates open
 * rather than tracing individual letterforms (there's no way to hand-author
 * stroke paths for an arbitrary dynamic name), which reads convincingly as
 * "being written" for a running-script face without needing one.
 *
 * Sized in SVG viewBox units measured from the text's own rendered
 * bounding box, then scaled to fill the container at width:100% — so a
 * short name and a long one both fill exactly the space they're given
 * instead of a fixed font-size overflowing or leaving it half-empty. */
export default function WinnerSignature({ name }) {
  const textRef = useRef(null);
  const clipRectRef = useRef(null);
  const [box, setBox] = useState(null);
  // getBBox() measures whatever font is actually painted right now — if
  // Pinyon Script hasn't finished loading yet, that's the fallback cursive
  // font, and the measured box won't match once the real face swaps in.
  const [fontReady, setFontReady] = useState(false);

  useEffect(() => {
    if (!document.fonts) {
      setFontReady(true);
      return;
    }
    document.fonts.load('92px "Pinyon Script"').finally(() => setFontReady(true));
  }, []);

  useLayoutEffect(() => {
    if (!textRef.current || !fontReady) return;
    const bbox = textRef.current.getBBox();
    setBox({
      width: bbox.width + bbox.x * 2,
      height: bbox.height + bbox.y * 2,
    });
  }, [name, fontReady]);

  useLayoutEffect(() => {
    if (!box || !clipRectRef.current) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      gsap.set(clipRectRef.current, { attr: { width: box.width } });
      return;
    }
    gsap.fromTo(
      clipRectRef.current,
      { attr: { width: 0 } },
      { attr: { width: box.width }, duration: 2, ease: "power2.inOut", delay: 0.15 }
    );
  }, [box]);

  return (
    <svg
      className="winner-signature"
      viewBox={box ? `0 0 ${box.width} ${box.height}` : "0 0 400 120"}
      width="100%"
      height="auto"
      role="img"
      aria-label={name}
    >
      <clipPath id="winner-signature-clip">
        <rect ref={clipRectRef} x="0" y="0" width="0" height={box ? box.height : 120} />
      </clipPath>
      <text
        ref={textRef}
        x="0"
        y={box ? box.height * 0.78 : 90}
        className="winner-signature-text"
        clipPath="url(#winner-signature-clip)"
      >
        {name}
      </text>
    </svg>
  );
}
