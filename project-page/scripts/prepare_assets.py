#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


POINT_CLOUDS = {
    "wok": "research/teaser_materials/02_wok_lid_handle/03_annotation_vis/gaussian_part_scores_colored.ply",
    "microwave": "research/teaser_materials/all_objects/49_microwave_oven_button/03_annotation_vis/gaussian_part_scores_colored.ply",
    "lamp": "research/teaser_materials/pipeline_figure/lamp_09/03_annotation_vis/gaussian_part_scores_colored.ply",
}

SAMPLES = {
    "wok": "research/teaser_materials/02_wok_lid_handle",
    "microwave": "research/teaser_materials/all_objects/49_microwave_oven_button",
    "phone": "research/teaser_materials/all_objects/15_cellular_telephone_power_button",
    "pan": "research/teaser_materials/all_objects/18_frying_pan_handle",
    "remote": "research/teaser_materials/all_objects/12_remote_control_channel_navigation_button",
    "toilet": "research/teaser_materials/all_objects/23_toilet_lid",
}

SAMPLE_FILES = {
    "source": "01_lvis_source.png",
    "geometry": "02_gaussian_render_hero.png",
    "annotation": "03_annotation_render_hero.png",
    "prediction": "04_model_prediction_heatmap.png",
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


def downsample_ascii_ply(source: Path, output: Path, max_points: int = 60_000) -> None:
    with source.open("r", encoding="ascii") as handle:
        header: list[str] = []
        vertex_count = 0
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"invalid PLY header: {source}")
            header.append(line)
            if line.startswith("format ") and "ascii" not in line:
                raise ValueError(f"only ASCII PLY input is supported: {source}")
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        if vertex_count <= 0:
            raise ValueError(f"missing vertex count: {source}")

        stride = max(1, (vertex_count + max_points - 1) // max_points)
        selected_count = (vertex_count + stride - 1) // stride
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as target:
            target.write(b"ply\n")
            target.write(b"format binary_little_endian 1.0\n")
            target.write(f"element vertex {selected_count}\n".encode("ascii"))
            target.write(b"property float x\nproperty float y\nproperty float z\n")
            target.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            written = 0
            for index in range(vertex_count):
                line = handle.readline()
                if index % stride:
                    continue
                values = line.split()
                if len(values) < 6:
                    raise ValueError(f"invalid vertex row {index}: {source}")
                xyz = (float(values[0]), float(values[1]), float(values[2]))
                rgb = (int(values[3]), int(values[4]), int(values[5]))
                target.write(struct.pack("<fffBBB", *xyz, *rgb))
                written += 1
            if written != selected_count:
                raise RuntimeError(f"expected {selected_count} points, wrote {written}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare static assets for the AffordAny project page")
    parser.add_argument("workspace_root", type=Path, help="Experimental AffordAny workspace containing paper_formal and research")
    args = parser.parse_args()
    workspace = args.workspace_root.resolve()
    output_root = Path(__file__).resolve().parents[1] / "public" / "assets"

    paper_output = output_root / "paper"
    pdf_to_webp(workspace / "paper_formal/figs/teaser.pdf", paper_output / "teaser.webp")
    pdf_to_webp(workspace / "paper_formal/figs/pipline.pdf", paper_output / "pipeline.webp")
    to_webp(workspace / "paper_formal/figs/overview_v3_7.drawio.png", paper_output / "architecture.webp", max_width=1700)
    to_webp(workspace / "paper_formal/figs/comparison_v2_red.png", paper_output / "comparison.webp", max_width=1900)

    for name, relative_path in POINT_CLOUDS.items():
        downsample_ascii_ply(workspace / relative_path, output_root / "pointclouds" / f"{name}.ply")

    for sample_id, relative_dir in SAMPLES.items():
        source_dir = workspace / relative_dir
        for output_name, source_name in SAMPLE_FILES.items():
            to_webp(source_dir / source_name, output_root / "samples" / sample_id / f"{output_name}.webp", max_width=900)

    print(f"Prepared project-page assets under {output_root}")


if __name__ == "__main__":
    main()
