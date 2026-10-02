import { Suspense, useEffect, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF, Environment, ContactShadows, Bounds, Html } from "@react-three/drei";
import { playSwitchSound } from "../lib/switchSound";
import "./CarViewer.css";

function CarModel(props) {
  const { scene } = useGLTF("/models/car.glb");
  // Disabling raycast on the top-level primitive only stops R3F testing the
  // group itself — it still recurses into every child mesh's own (default)
  // raycast method independently. With 33 materials worth of geometry, that
  // per-triangle intersection test running on every pointer move is what
  // actually read as a stutter; traversing and neutering it on each child is
  // the fix that removes the work rather than half-removing it.
  useEffect(() => {
    scene.traverse((obj) => {
      obj.raycast = () => null;
    });
  }, [scene]);
  return <primitive object={scene} {...props} />;
}
useGLTF.preload("/models/car.glb");

function Loader() {
  return (
    <Html center>
      <div className="car-loading spinner" />
    </Html>
  );
}

// Off by default: only the faint fill survives, so the car reads as a barely
// there silhouette until the bulb is switched on. On: the full rig that
// actually shows the model off, including a light hitting the floor. Every
// intensity lerps toward its target each frame rather than snapping, so
// flicking the switch reads as lights physically warming up, not a hard cut.
const LIGHTS_OFF = { ambient: 0.04, key: 0, fill: 0.12, rim: 0, ground: 0 };
const LIGHTS_ON = { ambient: 0.5, key: 1.4, fill: 0.5, rim: 0.6, ground: 1.1 };

function Lighting({ lightsOn }) {
  const ambientRef = useRef(null);
  const keyRef = useRef(null);
  const fillRef = useRef(null);
  const rimRef = useRef(null);
  const groundRef = useRef(null);

  useEffect(() => {
    // A SpotLight's target is a plain Object3D that isn't part of the scene
    // graph by default, so its matrixWorld never updates from a scene
    // traversal — without this it silently keeps aiming at the origin
    // regardless of the position set below.
    if (!groundRef.current) return;
    groundRef.current.target.position.set(0, -1, 0.5);
    groundRef.current.target.updateMatrixWorld();
  }, []);

  useFrame(() => {
    const target = lightsOn ? LIGHTS_ON : LIGHTS_OFF;
    const ease = 0.06;
    if (ambientRef.current) ambientRef.current.intensity += (target.ambient - ambientRef.current.intensity) * ease;
    if (keyRef.current) keyRef.current.intensity += (target.key - keyRef.current.intensity) * ease;
    if (fillRef.current) fillRef.current.intensity += (target.fill - fillRef.current.intensity) * ease;
    if (rimRef.current) rimRef.current.intensity += (target.rim - rimRef.current.intensity) * ease;
    if (groundRef.current) groundRef.current.intensity += (target.ground - groundRef.current.intensity) * ease;
  });

  return (
    <>
      <ambientLight ref={ambientRef} intensity={LIGHTS_OFF.ambient} />
      <directionalLight ref={keyRef} position={[6, 9, 4]} intensity={LIGHTS_OFF.key} castShadow />
      <directionalLight ref={fillRef} position={[-6, 3, -4]} intensity={LIGHTS_OFF.fill} color="#ffffff" />
      {/* The one accent colour, as the car's own rim glow — everything else
          in the rig is neutral white/grey. */}
      <directionalLight ref={rimRef} position={[0, 2, -6]} intensity={LIGHTS_OFF.rim} color="#c94a5c" />
      {/* Aimed straight down at the floor beneath the car, so the showroom
          reads as genuinely lit — not just the model picked out of black —
          with a visible pool of light on the ground, not just a contact
          shadow. */}
      <spotLight
        ref={groundRef}
        position={[0, 5, 0.5]}
        angle={0.55}
        penumbra={0.6}
        intensity={LIGHTS_OFF.ground}
        color="#ffffff"
      />
    </>
  );
}

export default function CarViewer({ className, lightsOn, onToggle }) {
  const controlsRef = useRef(null);

  const handleToggle = () => {
    playSwitchSound(!lightsOn);
    onToggle();
  };

  return (
    <div className={`car-viewer ${className || ""}`}>
      {/* resize.scroll defaults to true in R3F, which re-measures the canvas
          on every scroll event; combined with Bounds' own resize watching
          below, a scroll-triggered remeasurement that reports even a
          sub-pixel change re-runs the camera-fit animation mid-scroll — that
          repeated re-fit is what read as the model glitching while scrolling
          up and down. Neither watcher is needed: the container's actual size
          only changes on a real window resize, not on scroll. */}
      <Canvas camera={{ position: [4, 1.6, 5], fov: 35 }} dpr={[1, 1.5]} shadows resize={{ scroll: false }}>
        <Lighting lightsOn={lightsOn} />
        <Suspense fallback={<Loader />}>
          <Bounds fit clip margin={1.3}>
            <CarModel />
          </Bounds>
          {/* Stays mounted always — toggling it in and out would re-suspend
              inside the same boundary as the car model and flash the loader
              on every switch flick. Its intensity alone carries the off/on
              difference. */}
          <Environment preset="city" environmentIntensity={lightsOn ? 1 : 0.12} />
        </Suspense>
        <ContactShadows position={[0, -1, 0]} opacity={lightsOn ? 0.65 : 0.1} blur={2} far={4} />
        {/* Stays perfectly still until dragged — no autoRotate, and nothing
            here reacts to hover in any way (only the drag/momentum below).
            Low damping gives it real momentum, and the polar range is left
            at its full default (0 to PI) so a drag can spin it in any
            direction, not just around one locked axis. */}
        <OrbitControls
          ref={controlsRef}
          enableZoom={false}
          enablePan={false}
          enableDamping
          dampingFactor={0.02}
          rotateSpeed={0.85}
        />
      </Canvas>

      <div className={`bulb-glow ${lightsOn ? "is-on" : ""}`} aria-hidden="true" />
      <button
        type="button"
        className={`light-bulb ${lightsOn ? "is-on" : ""}`}
        onClick={handleToggle}
        aria-pressed={lightsOn}
        aria-label={lightsOn ? "Turn showroom lights off" : "Turn showroom lights on"}
      >
        <svg viewBox="0 0 24 24" className="light-bulb-icon" fill="none" aria-hidden="true">
          <path
            d="M12 3a6.5 6.5 0 0 0-3.7 11.84c.44.31.7.81.7 1.35V17a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1v-.81c0-.54.26-1.04.7-1.35A6.5 6.5 0 0 0 12 3z"
            stroke="currentColor"
            strokeWidth="1.5"
            fill="currentColor"
            fillOpacity="0.18"
          />
          <path d="M10 21h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
}
