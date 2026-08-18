# M10

Instruction-to-part semantic bridge for stage2 packaged objects.

This module reads `stage2_dataset/objects/<object_id>/package/part_instances.json`, uses only `candidate/masked_instance.png` as visual evidence, and asks a Gemini-compatible `generateContent` endpoint to write exactly 3 spoken user-to-robot commands for each packaged part.

## Inputs

Required object-level inputs:

- `source/source_meta.json`
- `package/part_instances.json`
- `candidate/masked_instance.png`

Dataset-level execution only processes objects that already contain a `visualization/` directory, which is treated as the signal that part labeling completed successfully.

## Outputs

Per object, the module writes:

- `instruction/part_instructions.json`

The JSON contains:

- `object_id`
- `object_name`
- `model_name`
- `instructions_per_part`
- `part_instances_path`
- `evidence_image_paths`
- `parts[]`, each with:
  - `part_sample_id`
  - `part_name`
  - `prompt`
  - `instructions` (length 3)

When running at dataset scope, the module also writes a summary file at the stage2 root:

- `instruction_bridge_summary.json`

## Environment

Expected environment variables:

- `GEMINI_BASE_URL`
- `GEMINI_API_KEY`

Fallback API key env:

- `GOOGLE_API_KEY`

Request format uses the Gemini-compatible native style:

- `POST /v1beta/models/{model}:generateContent?key=...`
- multimodal image inputs are sent as `inline_data`

Prompt intent:

- downstream task is human-to-robot interaction
- the generated text is meant to condition an affordance decoder
- each instruction should sound like a real command a person would say to a robot so the target part becomes the correct contact/manipulation region

## Example

Single object:

```bash
python research/pipeline/module_m10_instruction_bridge/instruction_bridge.py \
  --stage2-root research/pipeline/outputs/datasets/lvis_real/full_dataset_rerun_v3/stage2_dataset \
  --object-id object_000507
```

Full dataset:

```bash
python research/pipeline/module_m10_instruction_bridge/instruction_bridge.py \
  --stage2-root research/pipeline/outputs/datasets/lvis_real/full_dataset_rerun_v3/stage2_dataset
```
