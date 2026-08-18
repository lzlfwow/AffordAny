import { useEffect, useMemo, useState } from "react";
import {
  BookOpen,
  Box,
  Database,
  ExternalLink,
  Github,
  Layers3,
  ScanSearch,
} from "lucide-react";
import { heroCandidates, samples, stats, type Sample } from "./data";

const imageModes = ["source", "geometry", "annotation", "prediction"] as const;
type ImageMode = (typeof imageModes)[number];

const modeLabels: Record<ImageMode, string> = {
  source: "Source image",
  geometry: "3D reconstruction",
  annotation: "Part annotation",
  prediction: "AffordAny",
};

function SiteNav() {
  return (
    <nav className="site-nav" aria-label="Project navigation">
      <a className="wordmark" href="#top">AffordAny</a>
      <div>
        <a href="#overview">Overview</a>
        <a href="#dataset">Dataset</a>
        <a href="#method">Method</a>
        <a href="#results">Results</a>
        <a href="https://github.com/lzlfwow/AffordAny" target="_blank" rel="noreferrer">Code</a>
      </div>
    </nav>
  );
}

function HeroGallery() {
  return (
    <div className="hero-gallery" aria-label="Candidate AffordAny examples">
      {heroCandidates.map((item, index) => (
        <figure className="hero-case" key={item.id}>
          <div className="case-media">
            <div className="case-labels" aria-hidden="true">
              <span>Monocular RGB</span>
              <span>3D affordance</span>
            </div>
            <img
              src={item.pair}
              alt={`${item.label} source image and ${item.target} affordance prediction`}
              loading={index < 2 ? "eager" : "lazy"}
              decoding="async"
              fetchPriority={index === 0 ? "high" : "auto"}
            />
          </div>
          <figcaption>
            <strong>{item.label} · {item.target}</strong>
            <span>“{item.instruction}”</span>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

function PublicationHeader() {
  return (
    <header className="publication-header" id="top">
      <h1>AffordAny</h1>
      <p className="paper-title">
        VLM-Guided Open-World 3D Affordance Grounding<br />from a Monocular RGB Image
      </p>
      <div className="publication-links">
        <a href="#overview"><BookOpen size={18} /><span>Paper</span></a>
        <a href="https://github.com/lzlfwow/AffordAny" target="_blank" rel="noreferrer">
          <Github size={18} /><span>Code</span>
        </a>
      </div>
      <p className="tldr">
        <strong>TL;DR:</strong> AffordAny reconstructs an object from one real image and grounds a free-form interaction instruction directly on its 3D geometry.
      </p>
      <HeroGallery />
    </header>
  );
}

function AbstractSection() {
  return (
    <section className="narrow-section abstract-section">
      <h2>Abstract</h2>
      <p>
        Open-world 3D affordance grounding requires localizing functional object parts in 3D from free-form language. AffordAny builds large-scale text-conditioned 3D part supervision from monocular RGB images, grounds affordances with a frozen VLM-guided decoder, and improves open-world generalization through pseudo-label self-training. The resulting benchmark spans 5,334 objects, 10,633 validated parts, and 473 categories, while the decoder combines projected image evidence, instruction-conditioned semantic prototypes, and bidirectional geometry-semantic interaction.
      </p>
    </section>
  );
}

function PaperOverview() {
  return (
    <section className="content-section" id="overview">
      <div className="section-heading centered">
        <h2>Paper at a glance</h2>
        <p>From real-world images to open-world 3D interaction regions.</p>
      </div>
      <figure className="paper-figure original-figure">
        <img src="assets/paper/teaser.svg" alt="AffordAny paper teaser" loading="lazy" decoding="async" />
        <figcaption>The original teaser figure from the paper. <a href="assets/paper/teaser.pdf" target="_blank" rel="noreferrer">Open original PDF</a></figcaption>
      </figure>
    </section>
  );
}

const pipelineStages = [
  ["1", "LVIS instances", "Real images with long-tail category coverage"],
  ["2", "3D reconstruction", "Object-centric geometry from one RGB view"],
  ["3", "Multi-view rendering", "Canonical views expose functional surfaces"],
  ["4", "Part discovery", "Open-vocabulary parts and instructions"],
  ["5", "3D label lifting", "Multi-view masks become point supervision"],
];

function DatasetSection() {
  return (
    <section className="content-section dataset-overview" id="dataset">
      <div className="section-heading centered">
        <h2>AffordAny Dataset</h2>
        <p>Real images, reconstructed 3D objects, open-vocabulary parts, and free-form interaction instructions.</p>
      </div>
      <div className="dataset-stats">
        {stats.map(([value, label]) => <div key={label}><strong>{value}</strong><span>{label}</span></div>)}
      </div>
      <figure className="paper-figure pipeline-figure">
        <img src="assets/paper/pipeline.webp" alt="AffordAny dataset construction pipeline" loading="lazy" decoding="async" />
      </figure>
      <div className="pipeline-steps">
        {pipelineStages.map(([number, title, body]) => (
          <div key={number}><span>{number}</span><strong>{title}</strong><p>{body}</p></div>
        ))}
      </div>
    </section>
  );
}

function DatasetExplorer() {
  const [selectedId, setSelectedId] = useState(samples[0].id);
  const [imageMode, setImageMode] = useState<ImageMode>("prediction");
  const [split, setSplit] = useState("All splits");
  const visibleSamples = useMemo(
    () => split === "All splits" ? samples : samples.filter((sample) => sample.split === split),
    [split],
  );
  const selected = samples.find((sample) => sample.id === selectedId) ?? visibleSamples[0] ?? samples[0];

  useEffect(() => {
    if (!visibleSamples.some((sample) => sample.id === selectedId)) {
      setSelectedId(visibleSamples[0]?.id ?? samples[0].id);
    }
  }, [selectedId, visibleSamples]);

  return (
    <section className="content-section explorer-section" id="explorer">
      <div className="section-heading explorer-heading">
        <div><h2>Dataset Explorer</h2><p>Inspect source images, reconstructed geometry, labels, and model predictions.</p></div>
        <select value={split} onChange={(event) => setSplit(event.target.value)} aria-label="Evaluation split">
          <option>All splits</option>
          <option>Unseen instruction</option>
          <option>Unseen object</option>
          <option>Unseen category</option>
        </select>
      </div>
      <div className="sample-selector">
        {visibleSamples.map((sample) => (
          <button
            type="button"
            key={sample.id}
            className={sample.id === selected.id ? "active" : ""}
            onClick={() => setSelectedId(sample.id)}
          >
            <img src={sample.images.source} alt="" loading="lazy" decoding="async" />
            <span><strong>{sample.category}</strong><small>{sample.part}</small></span>
          </button>
        ))}
      </div>
      <div className="explorer-toolbar segmented light">
        {imageModes.map((item) => (
          <button type="button" key={item} className={item === imageMode ? "active" : ""} onClick={() => setImageMode(item)}>
            {modeLabels[item]}
          </button>
        ))}
      </div>
      <div className="explorer-main">
        <div className="sample-image-wrap">
          <img
            src={selected.images[imageMode]}
            alt={`${modeLabels[imageMode]} for ${selected.category}`}
            loading="lazy"
            decoding="async"
          />
        </div>
        <aside className="sample-meta">
          <span>{selected.split}</span>
          <h3>{selected.category}</h3>
          <dl>
            <div><dt>Target part</dt><dd>{selected.part}</dd></div>
            <div><dt>AffordAny IoU</dt><dd>{selected.iou.toFixed(3)}</dd></div>
          </dl>
          <blockquote>“{selected.instruction}”</blockquote>
        </aside>
      </div>
    </section>
  );
}

function MethodSection() {
  return (
    <section className="content-section" id="method">
      <div className="section-heading centered">
        <h2>VLM-Guided Affordance Decoder</h2>
        <p>Frozen visual-language features meet local 3D geometry through instruction-aware bidirectional reasoning.</p>
      </div>
      <figure className="paper-figure architecture-figure">
        <img src="assets/paper/architecture.webp" alt="AffordAny decoder architecture" loading="lazy" decoding="async" />
      </figure>
      <div className="method-notes">
        <div><Box /><p><strong>Projection injection</strong><br />Connects visible 3D points with image tokens.</p></div>
        <div><Layers3 /><p><strong>Semantic compression</strong><br />Builds instruction-aware semantic prototypes.</p></div>
        <div><ScanSearch /><p><strong>Bidirectional GPBlock</strong><br />Exchanges evidence between geometry and semantics.</p></div>
      </div>
    </section>
  );
}

function ResultsSection() {
  return (
    <section className="content-section" id="results">
      <div className="section-heading centered">
        <h2>Generalization</h2>
        <p>One evaluation protocol, three ways to be unseen: object, category, and instruction.</p>
      </div>
      <figure className="paper-figure original-figure comparison-original">
        <img
          src="assets/paper/comparison.svg"
          alt="Original AffordAny qualitative comparison figure"
          loading="lazy"
          decoding="async"
        />
        <figcaption>The original qualitative comparison figure from the paper. <a href="assets/paper/comparison.pdf" target="_blank" rel="noreferrer">Open original PDF</a></figcaption>
      </figure>
      <div className="result-summary">
        <div><Database /><span><strong>5,325</strong> disjoint pseudo-label objects</span></div>
        <div><Box /><span><strong>+6.3%</strong> unseen-category mIoU</span></div>
        <div><ScanSearch /><span><strong>p &lt; 0.01</strong> paired bootstrap</span></div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer>
      <div><strong>AffordAny</strong><p>VLM-guided open-world 3D affordance grounding from a monocular RGB image.</p></div>
      <div>
        <a href="https://github.com/lzlfwow/AffordAny" target="_blank" rel="noreferrer"><Github size={17} />Code<ExternalLink size={13} /></a>
        <span><BookOpen size={17} />Paper · arXiv pending</span>
      </div>
      <small>Code released under Apache-2.0.</small>
    </footer>
  );
}

export default function App() {
  return (
    <>
      <SiteNav />
      <main>
        <PublicationHeader />
        <AbstractSection />
        <PaperOverview />
        <DatasetSection />
        <DatasetExplorer />
        <MethodSection />
        <ResultsSection />
      </main>
      <Footer />
    </>
  );
}
