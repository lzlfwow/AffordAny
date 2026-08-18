# AffordAny Decoder

This directory contains the supervised AffordAny decoder used in the paper.
It consumes cached 3D point features, projected visual tokens, and VLM
instruction features, and predicts a point-level affordance heatmap.

## Structure

- `src/affordance_decoder_vlm_backbone_cosmos2b_v5/`: model, dataset, losses,
  metrics, projection utilities, and distributed helpers.
- `scripts/train.py`: configurable single- or multi-GPU training entry point.
- `scripts/evaluate.py`: checkpoint evaluation on a manifest.
- `scripts/train_v6_tuned.sh`: paper training configuration with caller-supplied
  cache and output paths.

## Commands

Run commands from the repository root:

```bash
bash models/affordance_decoder_vlm_backbone_cosmos2b_v5/scripts/train_v6_tuned.sh \
  /path/to/cache /path/to/output

python models/affordance_decoder_vlm_backbone_cosmos2b_v5/scripts/evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --manifest /path/to/manifest.json \
  --output-json /path/to/metrics.json
```

The cache format and released manifests will be documented on the AffordAny
dataset page.
