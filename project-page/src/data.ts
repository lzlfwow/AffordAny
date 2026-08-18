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
  pair: `assets/hero/${id}/pair.webp`,
  source: `assets/hero/${id}/source.webp`,
  prediction: `assets/hero/${id}/prediction.webp`,
}));
