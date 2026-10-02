import { useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, RoundedBox, MeshTransmissionMaterial, Sphere, Environment } from "@react-three/drei";
import "./Button3D.css";

/** A playful react-three-fiber button: a liquid-glass pill (drei's
 * MeshTransmissionMaterial) that warps harder on hover, orbited by a few
 * small bubbles that speed up and grow when hovered and read visibly through
 * the glass itself — the refraction has real colour to bend precisely
 * because those bubbles exist behind it, not just an empty canvas.
 *
 * Every animated value here is driven by a phase that accumulates via
 * per-frame delta, never by `elapsedTime * speed`. That distinction matters:
 * multiplying a large elapsed-time by a speed that steps between two values
 * on hover snaps the product (and therefore the orbit position / noise
 * phase) to a wildly different number the instant the hover state flips —
 * which is exactly the "cut" the animation used to make. Accumulating phase
 * means only the *rate* changes at that moment, not the position itself.
 *
 * The orbit radius and scale used to snap instantly on hover, though
 * (`orbitRadius * 1.35` / `scale.setScalar(1.4)` applied directly from the
 * boolean, no easing) — the angle stayed smooth but the *distance from
 * centre* popped straight to its new value the very next frame, which reads
 * exactly as "the ball jumped to a different spot." Both now lerp in
 * alongside the phase instead of snapping. */
function Bubble({ radius, orbitRadius, speed, offset, hovered }) {
  const ref = useRef(null);
  const phase = useRef(offset);
  const radiusMul = useRef(1);
  const scaleMul = useRef(1);
  useFrame((_, delta) => {
    if (!ref.current) return;
    phase.current += delta * (hovered ? speed * 2.2 : speed);
    radiusMul.current += ((hovered ? 1.35 : 1) - radiusMul.current) * 0.12;
    scaleMul.current += ((hovered ? 1.4 : 1) - scaleMul.current) * 0.12;
    const t = phase.current;
    const r = orbitRadius * radiusMul.current;
    ref.current.position.set(Math.cos(t) * r, Math.sin(t * 1.3) * 0.3, Math.sin(t) * r * 0.4);
    ref.current.scale.setScalar(scaleMul.current);
  });
  return (
    <Sphere ref={ref} args={[radius, 16, 16]}>
      <meshStandardMaterial color="#c94a5c" emissive="#c94a5c" emissiveIntensity={0.4} roughness={0.15} metalness={0.3} />
    </Sphere>
  );
}

function ButtonMesh({ hovered, pressed }) {
  const meshRef = useRef(null);
  useFrame(() => {
    if (!meshRef.current) return;
    const target = pressed ? 0.94 : hovered ? 1.08 : 1;
    meshRef.current.scale.lerp({ x: target, y: target, z: target }, 0.15);
  });
  return (
    <group>
      {/* Float's own speed stays fixed for the same reason as before (it
          derives its phase from elapsedTime * speed internally); only its
          intensities respond to hover. MeshTransmissionMaterial's "time"
          uniform is always raw elapsedTime, unscaled by anything — verified
          in its source before reaching for it, since MeshDistortMaterial's
          `speed` prop turned out to have exactly the jump-cut bug this
          component was built to avoid. distortion/temporalDistortion only
          scale the *magnitude* of the liquid warping, never its phase, so
          hover can safely animate them. */}
      <Float speed={2.4} rotationIntensity={0.35} floatIntensity={0.75}>
        <RoundedBox ref={meshRef} args={[3.4, 1.1, 0.7]} radius={0.5} smoothness={8}>
          <MeshTransmissionMaterial
            color="#f1c8ce"
            transmission={1}
            thickness={hovered ? 0.9 : 0.5}
            roughness={hovered ? 0.06 : 0.15}
            ior={1.35}
            chromaticAberration={0.06}
            distortion={hovered ? 0.55 : 0.25}
            distortionScale={0.4}
            temporalDistortion={hovered ? 0.25 : 0.08}
            envMapIntensity={1.3}
            samples={6}
            resolution={256}
          />
        </RoundedBox>
      </Float>
      {[0, 1, 2, 3].map((i) => (
        <Bubble key={i} radius={0.14 + (i % 2) * 0.06} orbitRadius={2.1 + i * 0.15} speed={0.4 + i * 0.12} offset={i * 1.6} hovered={hovered} />
      ))}
    </group>
  );
}

export default function Button3D({ children, onClick, disabled, className = "", tabIndex }) {
  const [hovered, setHovered] = useState(false);
  const [pressed, setPressed] = useState(false);

  return (
    <button
      className={`btn3d ${className}`}
      onClick={onClick}
      disabled={disabled}
      tabIndex={tabIndex}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => { setHovered(false); setPressed(false); }}
      onMouseDown={() => setPressed(true)}
      onMouseUp={() => setPressed(false)}
    >
      <Canvas className="btn3d-canvas" camera={{ position: [0, 0, 5], fov: 32 }} dpr={[1, 1.5]}>
        <ambientLight intensity={1.1} />
        <directionalLight position={[3, 4, 4]} intensity={2} />
        <directionalLight position={[-3, -2, 3]} intensity={0.9} color="#ffffff" />
        {/* Glass has nothing to refract without something to look through —
            gives the liquid material real reflections/highlights instead of
            reading flat. Same preset the car uses, for consistency. */}
        <Environment preset="city" />
        <ButtonMesh hovered={hovered && !disabled} pressed={pressed} />
      </Canvas>
      {/* Mirrors the 3D mesh's own scale.lerp target: without this the pill
          visibly grows/shrinks underneath a label that never moves, which is
          what read as static and disconnected from the button under it. */}
      <span
        className="btn3d-label"
        style={{ transform: `scale(${pressed ? 0.94 : hovered && !disabled ? 1.08 : 1})` }}
      >
        {children}
      </span>
    </button>
  );
}
