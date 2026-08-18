# AffordAny Self-Training

This directory contains the pseudo-label semi-supervised extension of the
AffordAny decoder.

The implementation adds confidence-weighted pseudo labels, uncertain-point
masking, real/pseudo loss scaling, curriculum scheduling, EMA evaluation, and
optional periodic pseudo-label refresh.

## Structure

- `src/affordance_decoder_selftraining_v1/`: mixed dataset and self-training
  losses.
- `scripts/build_mixed_manifest.py`: combines real and pseudo-label manifests.
- `scripts/train.py`: self-training entry point.
- `scripts/run_selftraining_round9.sh`: paper configuration with all local
  paths supplied by the caller.

## Command

Run from the repository root:

```bash
bash models/affordance_decoder_selftraining_v1/scripts/run_selftraining_round9.sh \
  /path/to/real_cache \
  /path/to/pseudo_cache \
  /path/to/initial_checkpoint.pt \
  /path/to/output
```

The script uses two GPUs by default. Override `CUDA_VISIBLE_DEVICES`,
`NUM_GPUS`, or `MASTER_PORT` in the environment when needed.
