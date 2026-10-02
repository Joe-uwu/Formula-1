/** A physical toggle-switch snap, synthesized with the Web Audio API — no
 * audio asset needed, and creating the AudioContext inside the click handler
 * satisfies the browser's autoplay gesture requirement for free.
 *
 * A real light switch doesn't read as a pitch (the old version here was a
 * pure square-wave blip, which sounds more like a synth chirp than a
 * switch); it reads as a short, broadband noise transient — the plastic
 * rocker's snap — plus a faint secondary contact click a beat later. On: a
 * brighter, higher-pass snap. Off: a duller, lower-pass thunk. */
export function playSwitchSound(isOn) {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const now = ctx.currentTime;

    const noiseDur = 0.045;
    const noiseBuffer = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * noiseDur), ctx.sampleRate);
    const data = noiseBuffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;

    const snap = (delay, freq, q, peak, dur) => {
      const src = ctx.createBufferSource();
      src.buffer = noiseBuffer;
      const filter = ctx.createBiquadFilter();
      filter.type = "bandpass";
      filter.frequency.value = freq;
      filter.Q.value = q;
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.001, now + delay);
      gain.gain.exponentialRampToValueAtTime(peak, now + delay + 0.002);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + delay + dur);
      src.connect(filter).connect(gain).connect(ctx.destination);
      src.start(now + delay);
      src.stop(now + delay + dur + 0.01);
      return src;
    };

    // The main throw, plus a fast pitch-drop tick giving it a plasticky body.
    snap(0, isOn ? 3200 : 1400, 0.9, isOn ? 0.5 : 0.4, noiseDur);

    const osc = ctx.createOscillator();
    const oscGain = ctx.createGain();
    osc.type = "square";
    if (isOn) {
      osc.frequency.setValueAtTime(1800, now);
      osc.frequency.exponentialRampToValueAtTime(600, now + 0.02);
    } else {
      osc.frequency.setValueAtTime(900, now);
      osc.frequency.exponentialRampToValueAtTime(220, now + 0.03);
    }
    oscGain.gain.setValueAtTime(0.001, now);
    oscGain.gain.exponentialRampToValueAtTime(0.14, now + 0.003);
    oscGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.035);
    osc.connect(oscGain).connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.04);

    // The mechanical rebound most physical toggles have a beat after the throw.
    snap(0.028, isOn ? 2400 : 1000, 1.1, 0.18, 0.02);

    osc.onended = () => ctx.close();
  } catch {
    // Audio is a nice-to-have here; never let it break the actual toggle.
  }
}
