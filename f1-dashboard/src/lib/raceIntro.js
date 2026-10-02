/** The racetrack boot-sequence: start lights, a car drawing itself along the
 * track, and a finish gate it crosses at the end. Shared by the full-screen
 * loading screen (which owns the animated 0-1 playback, once, before the app
 * mounts) and Hero (which settles straight to the story's resting frame,
 * since the loading screen already played the animation). Keeping the shape
 * of the track and the frame-by-frame math in one place means both always
 * agree on what "the same animation" looks like. */

export const TRACK_D =
  "M 80 700 C 300 700 380 320 620 320 C 820 320 880 620 1080 620 C 1260 620 1320 260 1520 200";
export const LIGHT_COUNT = 5;
export const LAUNCH = 0.05; // lights finish their sequence and go out — the car launches
export const DRAW_END = 0.88; // car reaches the finish gate

export const easeOutExpo = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));

/** Positions the finish gate at the track path's true end (needs real layout,
 * so call after the path node has mounted) and returns the path's total
 * length, which every other frame calculation needs. */
export function placeFinishGate(pathEl, flagEl) {
  const len = pathEl.getTotalLength();
  const end = pathEl.getPointAtLength(len);
  if (flagEl) flagEl.setAttribute("transform", `translate(${end.x} ${end.y})`);
  return len;
}

/** Drives the scene's DOM (imperatively — this runs every animation frame,
 * too often for React state) from a single 0-1 progress value. */
export function applyRaceFrame(progress, len, refs) {
  const { path, car, flag, lights, speedLines, speedVal, gearVal } = refs;

  const lightSeq = Math.min(1, progress / LAUNCH);
  (lights || []).forEach((lightEl, i) => {
    if (!lightEl) return;
    const litAt = i / LIGHT_COUNT;
    const isOut = progress >= LAUNCH;
    lightEl.style.fill = isOut ? "#1a1a1a" : lightSeq > litAt ? "#c94a5c" : "#2a2a2a";
    lightEl.style.filter = !isOut && lightSeq > litAt ? "drop-shadow(0 0 6px #c94a5c)" : "none";
  });

  const drawT = Math.min(1, Math.max(0, (progress - LAUNCH) / (DRAW_END - LAUNCH)));
  if (path && len) path.style.strokeDashoffset = String(len * (1 - drawT));
  if (car && path && len) {
    const dist = drawT * len;
    const pt = path.getPointAtLength(dist);
    const lookahead = path.getPointAtLength(Math.min(len, dist + 1));
    const angle = Math.atan2(lookahead.y - pt.y, lookahead.x - pt.x) * (180 / Math.PI);
    car.style.transform = `translate(${pt.x}px, ${pt.y}px) rotate(${angle}deg)`;
    car.style.opacity = progress >= LAUNCH ? "1" : "0";
  }

  if (speedLines) {
    const speedT = Math.min(1, Math.max(0, (progress - LAUNCH) / 0.5));
    speedLines.style.opacity = String(speedT * 0.45);
    speedLines.style.backgroundPosition = `${-progress * 2400}px 0`;
  }
  if (speedVal) {
    const kmh = Math.round(drawT * 341);
    speedVal.textContent = progress >= LAUNCH ? String(kmh) : "0";
  }
  if (gearVal) {
    const gear = progress < LAUNCH ? 0 : Math.min(8, 1 + Math.floor(drawT * 7.2));
    gearVal.textContent = progress < LAUNCH ? "N" : String(gear);
  }

  if (flag) {
    const flagT = Math.min(1, Math.max(0, (progress - (DRAW_END - 0.08)) / 0.1));
    flag.style.opacity = String(flagT);
  }
}
