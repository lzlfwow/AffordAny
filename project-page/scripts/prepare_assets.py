#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

SAMPLES = {
    "umbrella": "research/teaser_materials/all_objects/28_umbrella_canopy",
    "fork": "research/teaser_materials/all_objects/35_fork_tines",
    "phone": "research/teaser_materials/all_objects/15_cellular_telephone_power_button",
    "pan": "research/teaser_materials/all_objects/18_frying_pan_handle",
    "remote": "research/teaser_materials/all_objects/12_remote_control_channel_navigation_button",
    "glass": "research/teaser_materials/all_objects/24_glass_(drink_container)_body",
}

SAMPLE_FILES = {
    "source": "01_lvis_source.png",
    "geometry": "02_gaussian_render_hero.png",
    "annotation": "03_annotation_render_hero.png",
    "prediction": "04_model_prediction_heatmap.png",
}

HERO_CANDIDATES = {
    "tennis_racket": "research/teaser_materials/batch_6cats_x5/tennis_racket_04",
    "lamp": "research/teaser_materials/batch_6cats_x5/lamp_01",
    "steering_wheel": "research/teaser_materials/batch_6cats_x5/steering_wheel_04",
    "bottle": "research/teaser_materials/batch_6cats_x5/bottle_02",
    "phone": "research/teaser_materials/batch_6cats_x5/cellular_telephone_05",
    "hairbrush": "research/teaser_materials/batch_6cats_x5/hairbrush_02",
}


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def to_webp(source: Path, output: Path, max_width: int = 1800) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(source),
        "-vf", f"scale='min({max_width},iw)':-2", "-c:v", "libwebp",
        "-quality", "82", str(output),
    )


def pdf_to_webp(source: Path, output: Path, max_width: int = 1900) -> None:
    with tempfile.TemporaryDirectory(prefix="affordany-pdf-") as temp_dir:
        raster = Path(temp_dir) / "page"
        run("pdftoppm", "-png", "-r", "160", "-singlefile", str(source), str(raster))
        to_webp(raster.with_suffix(".png"), output, max_width=max_width)


def pdf_to_svg(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run("pdftocairo", "-svg", str(source), str(output))


def pair_webp(source: Path, prediction: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    panel = "scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:color=white"
    run(
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-i", str(prediction),
        "-filter_complex", f"[0:v]{panel}[source];[1:v]{panel}[prediction];[source][prediction]hstack",
        "-c:v", "libwebp", "-quality", "86", str(output),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare static assets for the AffordAny project page")
    parser.add_argument("workspace_root", type=Path, help="Experimental AffordAny workspace containing paper_formal and research")
    args = parser.parse_args()
    workspace = args.workspace_root.resolve()
    output_root = Path(__file__).resolve().parents[1] / "public" / "assets"

    paper_output = output_root / "paper"
    pdf_to_svg(workspace / "paper_formal/figs/teaser.pdf", paper_output / "teaser.svg")
    shutil.copy2(workspace / "paper_formal/figs/teaser.pdf", paper_output / "teaser.pdf")
    pdf_to_webp(workspace / "paper_formal/figs/pipline.pdf", paper_output / "pipeline.webp")
    to_webp(workspace / "paper_formal/figs/overview_v3_7.drawio.png", paper_output / "architecture.webp", max_width=1700)
    pdf_to_svg(workspace / "paper_formal/figs/comparison_v2_red.pdf", paper_output / "comparison.svg")
    shutil.copy2(workspace / "paper_formal/figs/comparison_v2_red.pdf", paper_output / "comparison.pdf")

    for sample_id, relative_dir in SAMPLES.items():
        source_dir = workspace / relative_dir
        for output_name, source_name in SAMPLE_FILES.items():
            to_webp(source_dir / source_name, output_root / "samples" / sample_id / f"{output_name}.webp", max_width=900)

    for sample_id, relative_dir in HERO_CANDIDATES.items():
        source_dir = workspace / relative_dir
        source_image = source_dir / "00_source_highlighted.png"
        prediction_image = source_dir / "04_model_prediction_heatmap.png"
        to_webp(
            source_image,
            output_root / "hero" / sample_id / "source.webp",
            max_width=900,
        )
        to_webp(
            prediction_image,
            output_root / "hero" / sample_id / "prediction.webp",
            max_width=900,
        )
        pair_webp(
            source_image,
            prediction_image,
            output_root / "hero" / sample_id / "pair.webp",
        )

    print(f"Prepared project-page assets under {output_root}")


if __name__ == "__main__":
    main()
