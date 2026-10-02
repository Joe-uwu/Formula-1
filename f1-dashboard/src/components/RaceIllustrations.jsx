/** Original hand-drawn SVG illustrations for the track-record reveal —
 * a trophy, a podium, and small hit/miss race badges — in place of stock or
 * AI-generated imagery (and specifically not real team liveries or driver
 * likenesses, which this project has no rights to depict). */

export function TrophyIllustration({ className }) {
  return (
    <svg className={className} viewBox="0 0 140 140" fill="none" aria-hidden="true">
      <path d="M45 30h50v28c0 14-11 25-25 25s-25-11-25-25V30z" fill="var(--series-1)" />
      <path d="M45 34c-10 0-18 8-18 16s6 14 14 15" stroke="var(--text-muted)" strokeWidth="4" strokeLinecap="round" fill="none" />
      <path d="M95 34c10 0 18 8 18 16s-6 14-14 15" stroke="var(--text-muted)" strokeWidth="4" strokeLinecap="round" fill="none" />
      <rect x="64" y="82" width="12" height="18" fill="var(--series-1)" />
      <rect x="50" y="100" width="40" height="10" rx="2" fill="var(--text-muted)" />
      <rect x="42" y="110" width="56" height="10" rx="2" fill="var(--text-primary)" />
      <circle cx="70" cy="46" r="9" fill="var(--surface-0)" opacity="0.5" />
    </svg>
  );
}

export function PodiumIllustration({ className }) {
  return (
    <svg className={className} viewBox="0 0 200 140" fill="none" aria-hidden="true">
      <rect x="8" y="60" width="54" height="60" fill="var(--series-2)" />
      <text x="35" y="98" textAnchor="middle" fontSize="28" fontWeight="900" fill="var(--surface-1)" fontFamily="var(--font-display)">2</text>
      <rect x="73" y="30" width="54" height="90" fill="var(--series-1)" />
      <text x="100" y="82" textAnchor="middle" fontSize="32" fontWeight="900" fill="var(--surface-1)" fontFamily="var(--font-display)">1</text>
      <rect x="138" y="78" width="54" height="42" fill="var(--baseline)" />
      <text x="165" y="106" textAnchor="middle" fontSize="24" fontWeight="900" fill="var(--surface-1)" fontFamily="var(--font-display)">3</text>
    </svg>
  );
}

export function HitBadge({ className }) {
  const sq = 8;
  return (
    <svg className={className} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      {Array.from({ length: 4 }).map((_, row) =>
        Array.from({ length: 4 }).map((_, col) => (
          <rect
            key={`${row}-${col}`}
            x={col * sq}
            y={row * sq}
            width={sq}
            height={sq}
            fill={(row + col) % 2 === 0 ? "var(--good)" : "var(--surface-1)"}
          />
        ))
      )}
    </svg>
  );
}

export function MissBadge({ className }) {
  return (
    <svg className={className} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <circle cx="16" cy="16" r="14" stroke="var(--baseline)" strokeWidth="3" />
      <path d="M11 11l10 10M21 11l-10 10" stroke="var(--text-muted)" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}
