import { useEffect, useMemo, useState } from "react";
import {
  ArrowDown,
  ArrowRight,
  Box,
  Braces,
  ChevronRight,
  Database,
  ExternalLink,
  FileText,
  Github,
  Layers3,
  Menu,
  ScanSearch,
  X,
} from "lucide-react";
import PointCloudViewer, { type ViewerMode } from "./PointCloudViewer";
import { heroObjects, samples, stats, type Sample } from "./data";

const imageModes = ["source", "geometry", "annotation", "prediction"] as const;
type ImageMode = (typeof imageModes)[number];

const modeLabels: Record<ImageMode, string> = {
  source: "Source image",
  geometry: "3D reconstruction",
  annotation: "Part annotation",
  prediction: "AffordAny",
};

function Header() {
  const [open, setOpen] = useState(false);
  const links = [
    ["Pipeline", "#pipeline"],
    ["Dataset", "#dataset"],
    ["Decoder", "#decoder"],
    ["Results", "#results"],
  ];
  return (
    <header className="site-header">
      <a href="#top" className="brand" aria-label="AffordAny home"><span className="brand-mark" />AffordAny</a>
      <nav className={open ? "nav-links open" : "nav-links"}>
        {links.map(([label, href]) => <a href={href} key={href} onClick={() => setOpen(false)}>{label}</a>)}
        <a href="https://github.com/lzlfwow/AffordAny" target="_blank" rel="noreferrer"><Github size={16} />Code</a>
      </nav>
      <button className="menu-button" type="button" onClick={() => setOpen((value) => !value)} aria-label="Toggle navigation">
        {open ? <X /> : <Menu />}
      </button>
    </header>
  );
}

function Hero() {
  const [objectIndex, setObjectIndex] = useState(0);
  const [partIndex, setPartIndex] = useState(1);
  const [mode, setMode] = useState<ViewerMode>("affordance");
  const object = heroObjects[objectIndex];
  const part = object.parts[Math.min(partIndex, object.parts.length - 1)];

  useEffect(() => setPartIndex(object.id === "wok" ? 1 : 0), [object.id]);

  return (
    <section className="hero" id="top">
      <PointCloudViewer object={object} part={part} mode={mode} />
      <div className="hero-copy">
        <p className="eyebrow">Open-world 3D affordance grounding</p>
        <h1>AffordAny</h1>
        <p className="hero-subtitle">VLM-guided functional grounding from a single monocular RGB image.</p>
        <div className="hero-actions">
          <a className="button primary" href="#paper"><FileText size={17} />Paper</a>
          <a className="button secondary" href="https://github.com/lzlfwow/AffordAny" target="_blank" rel="noreferrer"><Github size={17} />Code</a>
        </div>
      </div>
      <div className="hero-controls">
        <div className="control-block object-control">
          <span className="control-label">Object</span>
          <div className="segmented dark">
            {heroObjects.map((item, index) => (
              <button className={index === objectIndex ? "active" : ""} type="button" key={item.id} onClick={() => setObjectIndex(index)}>{item.label}</button>
            ))}
          </div>
        </div>
        <div className="instruction-control">
          <span className="control-label">Instruction</span>
          <div className="instruction-row">
            <p>“{part.instruction}”</p>
            <select value={part.name} onChange={(event) => setPartIndex(object.parts.findIndex((item) => item.name === event.target.value))} aria-label="Target part">
              {object.parts.map((item) => <option value={item.name} key={item.name}>{item.label}</option>)}
            </select>
          </div>
        </div>
        <div className="control-block mode-control">
          <span className="control-label">View</span>
          <div className="segmented dark">
            {(["geometry", "parts", "affordance"] as ViewerMode[]).map((item) => (
              <button className={mode === item ? "active" : ""} type="button" key={item} onClick={() => setMode(item)}>{item === "affordance" ? "Prediction" : item[0].toUpperCase() + item.slice(1)}</button>
            ))}
          </div>
        </div>
      </div>
      <div className="hero-result"><span>{object.split}</span><strong>IoU {object.iou}</strong></div>
      <a href="#overview" className="scroll-cue" aria-label="Continue"><ArrowDown size={18} /></a>
    </section>
  );
}

function StatsBand() {
  return <div className="stats-band">{stats.map(([value, label]) => <div key={label}><strong>{value}</strong><span>{label}</span></div>)}</div>;
}

function PaperOverview() {
  return (
    <section className="paper-overview section" id="overview">
      <div className="section-heading compact">
        <p className="eyebrow">Paper at a glance</p>
        <h2>From real images to open-world interaction regions</h2>
      </div>
      <figure className="paper-figure wide-figure">
        <img src="assets/paper/teaser.webp" alt="AffordAny paper teaser showing dataset construction and generalization examples" />
      </figure>
    </section>
  );
}

const pipelineStages = [
  ["01", "LVIS source", "Real-image instances with long-tail object coverage"],
  ["02", "3D reconstruction", "Object-centric geometry from one RGB observation"],
  ["03", "Multi-view rendering", "Six canonical views expose functional surfaces"],
  ["04", "Part discovery", "Open-vocabulary parts and interaction instructions"],
  ["05", "3D label lifting", "Multi-view voting fuses masks into 3D supervision"],
];

function PipelineSection() {
  return (
    <section className="pipeline-section section" id="pipeline">
      <div className="section-heading split-heading">
        <div><p className="eyebrow">Dataset construction</p><h2>One image becomes structured 3D supervision</h2></div>
        <p>The pipeline converts LVIS instances into validated object–part–instruction triples without requiring pre-built 3D assets.</p>
      </div>
      <figure className="paper-figure pipeline-figure"><img src="assets/paper/pipeline.webp" alt="AffordAny dataset construction pipeline from LVIS to 3D label lifting" /></figure>
      <div className="pipeline-steps">
        {pipelineStages.map(([number, title, body]) => <div className="pipeline-step" key={number}><span>{number}</span><h3>{title}</h3><p>{body}</p></div>)}
      </div>
    </section>
  );
}

function DatasetExplorer() {
  const [selectedId, setSelectedId] = useState(samples[0].id);
  const [imageMode, setImageMode] = useState<ImageMode>("prediction");
  const [split, setSplit] = useState("All splits");
  const visibleSamples = useMemo(() => split === "All splits" ? samples : samples.filter((sample) => sample.split === split), [split]);
  const selected = samples.find((sample) => sample.id === selectedId) ?? visibleSamples[0] ?? samples[0];

  useEffect(() => {
    if (!visibleSamples.some((sample) => sample.id === selectedId)) setSelectedId(visibleSamples[0]?.id ?? samples[0].id);
  }, [selectedId, visibleSamples]);

  return (
    <section className="dataset-section section" id="dataset">
      <div className="section-heading split-heading">
        <div><p className="eyebrow">Dataset explorer</p><h2>Inspect the supervision, not only the statistics</h2></div>
        <select value={split} onChange={(event) => setSplit(event.target.value)} aria-label="Evaluation split">
          <option>All splits</option><option>Unseen instruction</option><option>Unseen object</option><option>Unseen category</option>
        </select>
      </div>
      <div className="explorer-layout">
        <div className="sample-rail">
          {visibleSamples.map((sample) => (
            <button type="button" key={sample.id} className={sample.id === selected.id ? "sample-item active" : "sample-item"} onClick={() => setSelectedId(sample.id)}>
              <img src={sample.images.source} alt="" />
              <span><strong>{sample.category}</strong><small>{sample.part}</small></span>
              <ChevronRight size={17} />
            </button>
          ))}
        </div>
        <div className="sample-viewer">
          <div className="sample-image-wrap"><img src={selected.images[imageMode]} alt={`${modeLabels[imageMode]} for ${selected.category}`} /></div>
          <div className="segmented light image-tabs">
            {imageModes.map((item) => <button type="button" key={item} className={item === imageMode ? "active" : ""} onClick={() => setImageMode(item)}>{modeLabels[item]}</button>)}
          </div>
        </div>
        <aside className="sample-meta">
          <span className="split-label">{selected.split}</span>
          <h3>{selected.category}</h3>
          <dl><div><dt>Target part</dt><dd>{selected.part}</dd></div><div><dt>AffordAny IoU</dt><dd>{selected.iou.toFixed(3)}</dd></div></dl>
          <blockquote>“{selected.instruction}”</blockquote>
          <span className="sample-id">AFF-{selected.id.toUpperCase()}</span>
        </aside>
      </div>
    </section>
  );
}

function DecoderSection() {
  return (
    <section className="decoder-section section" id="decoder">
      <div className="section-heading split-heading">
        <div><p className="eyebrow">Affordance decoder</p><h2>Geometry and VLM semantics meet in both directions</h2></div>
        <p>A frozen Cosmos-2B backbone supplies visual and instruction features. Semantic compression and GPBlock preserve fine 3D structure while exchanging global interaction context.</p>
      </div>
      <div className="architecture-layout">
        <div className="architecture-points">
          <div><Braces /><span><strong>Projection injection</strong> connects image tokens to visible 3D points.</span></div>
          <div><Layers3 /><span><strong>Semantic compression</strong> condenses VLM features into instruction-aware prototypes.</span></div>
          <div><ScanSearch /><span><strong>Bidirectional GPBlock</strong> groups points and returns localized evidence.</span></div>
        </div>
        <figure className="paper-figure architecture-figure"><img src="assets/paper/architecture.webp" alt="AffordAny decoder architecture from the paper" /></figure>
      </div>
    </section>
  );
}

const metricRows = [
  ["AffordAny", ".428", ".305", ".680"],
  ["LASO", ".418", ".266", ".473"],
  ["LMAffordance3D", ".395", ".244", ".312"],
  ["OpenAD", ".346", ".246", ".517"],
];

function ResultsSection() {
  const [tab, setTab] = useState<Sample["split"]>("Unseen instruction");
  const examples = samples.filter((sample) => sample.split === tab).slice(0, 2);
  return (
    <section className="results-section section" id="results">
      <div className="section-heading"><p className="eyebrow">Generalization</p><h2>One protocol, three ways to be unseen</h2></div>
      <div className="segmented light result-tabs">
        {(["Unseen instruction", "Unseen object", "Unseen category"] as Sample["split"][]).map((item) => <button type="button" key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>{item}</button>)}
      </div>
      <div className="result-layout">
        <div className="result-examples">
          {examples.map((sample) => <figure key={sample.id}><img src={sample.images.prediction} alt={`AffordAny prediction for ${sample.category}`} /><figcaption><span>{sample.category} / {sample.part}</span><strong>IoU {sample.iou.toFixed(2)}</strong></figcaption></figure>)}
        </div>
        <div className="metrics-panel">
          <div className="metric-header"><span>Method</span><span>Object</span><span>Category</span><span>Instruction</span></div>
          {metricRows.map((row, index) => <div className={index === 0 ? "metric-row ours" : "metric-row"} key={row[0]}>{row.map((cell) => <span key={cell}>{cell}</span>)}</div>)}
          <small>IoU on formal evaluation splits</small>
        </div>
      </div>
      <figure className="paper-figure comparison-figure"><img src="assets/paper/comparison.webp" alt="Qualitative comparison from the AffordAny paper" /></figure>
      <div className="self-training-band">
        <div><Database /><span><strong>5,325</strong> disjoint pseudo-label objects</span></div>
        <ArrowRight />
        <div><Box /><span><strong>+6.3%</strong> unseen-category mIoU</span></div>
        <ArrowRight />
        <div><ScanSearch /><span><strong>p &lt; 0.01</strong> paired bootstrap</span></div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer id="paper">
      <div><span className="brand footer-brand"><span className="brand-mark" />AffordAny</span><p>VLM-guided open-world 3D affordance grounding from a monocular RGB image.</p></div>
      <div className="footer-links"><a href="https://github.com/lzlfwow/AffordAny" target="_blank" rel="noreferrer"><Github />Code<ExternalLink size={14} /></a><span><FileText />Paper · arXiv pending</span></div>
      <p className="license">Code released under Apache-2.0.</p>
    </footer>
  );
}

export default function App() {
  return <><Header /><main><Hero /><StatsBand /><PaperOverview /><PipelineSection /><DatasetExplorer /><DecoderSection /><ResultsSection /></main><Footer /></>;
}
