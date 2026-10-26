import { useEffect, useRef, useState } from "react";
import "./SiteFooter.css";

const EMAIL = "josephprasanna987@gmail.com";
const LINKEDIN = "https://www.linkedin.com/in/joseph-prasanna-1aba592a3";

/** The finish line: a chequered strip the hero's car crosses once the
 * footer scrolls into view, a timing-tower row for the credit, and the
 * contact links as team-radio channels. */
export default function SiteFooter() {
  const ref = useRef(null);
  const [crossed, setCrossed] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || !("IntersectionObserver" in window)) {
      setCrossed(true);
      return;
    }
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setCrossed(true);
        io.disconnect();
      }
    }, { threshold: 0.35 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <footer ref={ref} className={`ft${crossed ? " ft-crossed" : ""}`}>
      <div className="ft-flag" aria-hidden="true">
        <svg className="ft-car" viewBox="-34 -16 60 32" width="60" height="32">
          <rect x="-26" y="-9" width="52" height="18" rx="6" className="ft-car-body" />
          <rect x="16" y="-14" width="6" height="28" rx="1" className="ft-car-wing" />
          <rect x="-32" y="-11" width="8" height="22" rx="1" className="ft-car-wing" />
          <circle cx="-15" cy="-12" r="5.5" className="ft-car-wheel" />
          <circle cx="-15" cy="12" r="5.5" className="ft-car-wheel" />
          <circle cx="11" cy="-12" r="5.5" className="ft-car-wheel" />
          <circle cx="11" cy="12" r="5.5" className="ft-car-wheel" />
          <ellipse cx="-3" cy="0" rx="8" ry="5" className="ft-car-cockpit" />
        </svg>
      </div>

      <div className="ft-inner">
        <div className="ft-headline">
          <span className="ft-kicker">Lap 58 / 58</span>
          <p className="ft-title">That's the<br />chequered flag.</p>
        </div>

        <div className="ft-side">
          <div className="ft-tower" role="group" aria-label="Credits">
            <span className="ft-pos">P1</span>
            <span className="ft-bar" aria-hidden="true" />
            <span className="ft-driver">
              <span className="ft-driver-label">Created by</span>
              <span className="ft-driver-name">Joseph Suresh</span>
            </span>
            <span className="ft-time">FIN</span>
          </div>

          <div className="ft-radio">
            <a className="ft-channel" href={`mailto:${EMAIL}`}>
              <span className="ft-channel-label">Team radio · Email</span>
              <span className="ft-channel-value">{EMAIL}</span>
              <span className="ft-arrow" aria-hidden="true">↗</span>
            </a>
            <a className="ft-channel" href={LINKEDIN} target="_blank" rel="noopener noreferrer">
              <span className="ft-channel-label">Pit wall · LinkedIn</span>
              <span className="ft-channel-value">joseph-prasanna</span>
              <span className="ft-arrow" aria-hidden="true">↗</span>
            </a>
          </div>
        </div>
      </div>

      <div className="ft-base">
        <span>© {new Date().getFullYear()} Formula 1 Predictor</span>
        <span>Not affiliated with Formula 1. No pole-sitters were harmed.</span>
      </div>

      <div className="ft-ghost" aria-hidden="true">FIN</div>
    </footer>
  );
}
