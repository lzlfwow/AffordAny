# M11 DIT Part Segmentation

This module adds a Seed3D-2.0-style part segmentation entry point for Stage2 objects.
The current M6-M8 path builds part labels from 2D masks projected to Gaussian points;
M11 instead treats part segmentation as an object-level model output that can be
produced by an external PartSeg/PartDiT implementation.

## Expected external model

Seed3D 2.0 describes a two-stage part-level pipeline: `Seed3D-PartSeg` produces
functional surface regions from a mesh/point cloud, then `Seed3D-PartDiT` completes
and composes part-aware geometry. The paper does not include public inference code or
weights in this repository, so this module exposes a command-template interface.

Provide an executable command that writes:

- `part_membership_scores.npz` with keys `part_names`, `scores`, `best_part_index`,
  `max_scores`, `visible_counts`.
- `unknown_mask.npz` with key `unknown_mask` and optional diagnostic masks.
- Optional `part_segments.ply` or per-part geometry under `dit_partseg/`.

The command template can use these placeholders:

- `{object_dir}`
- `{gaussian_path}`
- `{prompt_path}`
- `{output_dir}`
- `{part_membership_path}`
- `{unknown_mask_path}`
- `{part_segments_path}`
- `{meta_path}`

## Local adapter

For pipeline integration tests and for inspecting existing objects, the default local
executor adapts existing `label3d/part_membership_scores.npz` and
`label3d/unknown_mask.npz` into the new `dit_partseg/` layout. This is not a DIT model;
it is a compatibility adapter that keeps downstream packaging format stable until the
external Seed3D-style model is supplied.

## Example

To run an external model:

```python
from research.pipeline.module_m11_dit_part_segmentation import (
    DitPartSegmentationConfig,
    ExternalDitPartSegmentationExecutor,
    build_dit_part_segmentation_request,
    execute_dit_part_segmentation,
)

request = build_dit_part_segmentation_request(object_dir)
config = DitPartSegmentationConfig(
    backend="external_seed3d_partdit",
    command_template=(
        "python tools/infer_partseg.py "
        "--mesh {gaussian_path} --prompts {prompt_path} --out {output_dir}"
    ),
)
execute_dit_part_segmentation(
    request,
    config=config,
    executor=ExternalDitPartSegmentationExecutor(),
)
```
