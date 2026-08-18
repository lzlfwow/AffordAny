export type Part = {
  name: string;
  label: string;
  instruction: string;
  color: [number, number, number];
};

export type HeroObject = {
  id: string;
  label: string;
  ply: string;
  split: string;
  iou: string;
  parts: Part[];
};

export const heroObjects: HeroObject[] = [
  {
    id: "wok",
    label: "Wok",
    ply: "assets/pointclouds/wok.ply",
    split: "Unseen instruction",
    iou: "0.86",
    parts: [
      { name: "handle", label: "Handle", instruction: "Pick up the wok by its side handle.", color: [191, 219, 254] },
      { name: "lid_handle", label: "Lid handle", instruction: "Hold the center knob so you can take the lid off.", color: [60, 210, 160] },
      { name: "lid", label: "Lid", instruction: "Lift the cover away from the pan.", color: [226, 85, 171] },
    ],
  },
  {
    id: "microwave",
    label: "Microwave",
    ply: "assets/pointclouds/microwave.ply",
    split: "Unseen instruction",
    iou: "0.83",
    parts: [
      { name: "door", label: "Door", instruction: "Pull the front panel to open the oven.", color: [126, 151, 174] },
      { name: "button", label: "Button", instruction: "Press the control to start heating.", color: [177, 171, 163] },
    ],
  },
  {
    id: "lamp",
    label: "Lamp",
    ply: "assets/pointclouds/lamp.ply",
    split: "Unseen instruction",
    iou: "0.79",
    parts: [
      { name: "base", label: "Base", instruction: "Steady the base so the lamp does not wobble.", color: [221, 235, 249] },
      { name: "lamp_head", label: "Lamp head", instruction: "Aim the light using the lamp head.", color: [173, 184, 194] },
      { name: "arm", label: "Arm", instruction: "Move the arm to adjust the light.", color: [126, 217, 87] },
    ],
  },
];

export type Sample = {
  id: string;
  category: string;
  part: string;
  instruction: string;
  split: "Unseen instruction" | "Unseen object" | "Unseen category";
  iou: number;
  images: { source: string; geometry: string; annotation: string; prediction: string };
};

const sampleImages = (id: string) => ({
  source: `assets/samples/${id}/source.webp`,
  geometry: `assets/samples/${id}/geometry.webp`,
  annotation: `assets/samples/${id}/annotation.webp`,
  prediction: `assets/samples/${id}/prediction.webp`,
});

export const samples: Sample[] = [
  { id: "wok", category: "Wok", part: "Lid handle", instruction: "Hold the center knob so you can take the lid off.", split: "Unseen instruction", iou: 0.855, images: sampleImages("wok") },
  { id: "microwave", category: "Microwave oven", part: "Button", instruction: "Press the control to start heating.", split: "Unseen instruction", iou: 0.833, images: sampleImages("microwave") },
  { id: "phone", category: "Cellular telephone", part: "Power button", instruction: "Press the power button.", split: "Unseen instruction", iou: 0.952, images: sampleImages("phone") },
  { id: "pan", category: "Frying pan", part: "Handle", instruction: "Pick up the pan by its grip.", split: "Unseen object", iou: 0.857, images: sampleImages("pan") },
  { id: "remote", category: "Remote control", part: "Navigation button", instruction: "Press the channel button.", split: "Unseen category", iou: 0.688, images: sampleImages("remote") },
  { id: "toilet", category: "Toilet", part: "Lid", instruction: "Close the toilet lid.", split: "Unseen category", iou: 0.736, images: sampleImages("toilet") },
];

export const stats = [
  ["5,334", "Objects"],
  ["10,633", "Validated parts"],
  ["31,899", "Instructions"],
  ["473", "Categories"],
  ["678", "Part types"],
];

export const heroCandidates = [
  ["tennis_racket", "Tennis racket", "Handle", "Hold the racket by its handle."],
  ["lamp", "Table lamp", "Base", "Steady the lamp by its base."],
  ["steering_wheel", "Steering wheel", "Wheel", "Turn the wheel to steer."],
  ["bottle", "Bottle", "Neck", "Hold the bottle at its neck."],
  ["phone", "Cellular telephone", "Screen", "Tap the screen to use the phone."],
  ["hairbrush", "Hairbrush", "Handle", "Grip the brush by its handle."],
].map(([id, label, target, instruction]) => ({
  id,
  label,
  target,
  instruction,
  source: `assets/hero/${id}/source.webp`,
  prediction: `assets/hero/${id}/prediction.webp`,
}));
