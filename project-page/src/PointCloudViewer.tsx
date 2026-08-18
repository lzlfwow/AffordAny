import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useLoader } from "@react-three/fiber";
import { AdaptiveDpr, OrbitControls } from "@react-three/drei";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import type { BufferAttribute, Group } from "three";
import { Color } from "three";
import { Maximize2, RotateCcw } from "lucide-react";
import type { HeroObject, Part } from "./data";

export type ViewerMode = "geometry" | "parts" | "affordance";

function heatColor(confidence: number): Color {
  const dark = new Color("#68716e");
  const red = new Color("#d92d20");
  const yellow = new Color("#ffd84a");
  if (confidence <= 0.03) return dark;
  if (confidence < 0.72) return dark.lerp(red, confidence / 0.72);
  return red.lerp(yellow, (confidence - 0.72) / 0.28);
}

function Cloud({ object, part, mode }: { object: HeroObject; part: Part; mode: ViewerMode }) {
  const loaded = useLoader(PLYLoader, `${import.meta.env.BASE_URL}${object.ply}`);
  const group = useRef<Group>(null);
  const geometry = useMemo(() => {
    const copy = loaded.clone();
    copy.computeBoundingSphere();
    copy.center();
    copy.computeBoundingSphere();
    return copy;
  }, [loaded]);
  const originalColors = useMemo(() => {
    const color = geometry.getAttribute("color") as BufferAttribute;
    return new Float32Array(color.array as ArrayLike<number>);
  }, [geometry]);
  const radius = geometry.boundingSphere?.radius || 1;

  useEffect(() => {
    const color = geometry.getAttribute("color") as BufferAttribute;
    const target = part.color.map((value) => value / 255);
    const neutral = new Color("#9aa3a0");
    for (let i = 0; i < color.count; i += 1) {
      const offset = i * 3;
      const r = originalColors[offset];
      const g = originalColors[offset + 1];
      const b = originalColors[offset + 2];
      if (mode === "parts") {
        color.setXYZ(i, r, g, b);
      } else if (mode === "geometry") {
        const shade = 0.78 + ((i * 17) % 23) / 150;
        color.setXYZ(i, neutral.r * shade, neutral.g * shade, neutral.b * shade);
      } else {
        const distance = Math.sqrt((r - target[0]) ** 2 + (g - target[1]) ** 2 + (b - target[2]) ** 2);
        const confidence = Math.max(0, 1 - distance / 0.48);
        const mapped = heatColor(confidence);
        color.setXYZ(i, mapped.r, mapped.g, mapped.b);
      }
    }
    color.needsUpdate = true;
  }, [geometry, mode, originalColors, part]);

  useFrame((_, delta) => {
    if (group.current) group.current.rotation.y += delta * 0.09;
  });

  return (
    <group ref={group} position={[0.82, -0.05, 0]} scale={1.38 / radius} rotation={[-0.2, 0.45, 0]}>
      <points geometry={geometry}>
        <pointsMaterial vertexColors size={0.011 * radius} sizeAttenuation transparent opacity={0.96} />
      </points>
    </group>
  );
}

function Loader() {
  return <div className="viewer-loading">Loading point cloud</div>;
}

export default function PointCloudViewer({ object, part, mode }: { object: HeroObject; part: Part; mode: ViewerMode }) {
  const [cameraKey, setCameraKey] = useState(0);

  const toggleFullscreen = () => {
    const element = document.querySelector(".hero-stage");
    if (!document.fullscreenElement) void element?.requestFullscreen();
    else void document.exitFullscreen();
  };

  return (
    <div className="hero-stage">
      <Suspense fallback={<Loader />}>
        <Canvas
          key={cameraKey}
          camera={{ position: [0, 0.1, 4.6], fov: 42 }}
          dpr={[1, 1.75]}
          gl={{ alpha: false, antialias: true, preserveDrawingBuffer: true }}
        >
          <color attach="background" args={["#111311"]} />
          <fog attach="fog" args={["#111311", 4.8, 7.5]} />
          <Cloud object={object} part={part} mode={mode} />
          <OrbitControls enablePan={false} minDistance={2.8} maxDistance={7} />
          <AdaptiveDpr pixelated />
        </Canvas>
      </Suspense>
      <div className="viewer-tools">
        <button type="button" className="icon-button" onClick={() => setCameraKey((value) => value + 1)} title="Reset view" aria-label="Reset view">
          <RotateCcw size={18} />
        </button>
        <button type="button" className="icon-button" onClick={toggleFullscreen} title="Fullscreen" aria-label="Fullscreen">
          <Maximize2 size={18} />
        </button>
      </div>
    </div>
  );
}
