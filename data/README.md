# Data

The AffordAny release separates source data, generated annotations, model
caches, and checkpoints from the Git repository.

## Local layout

```text
data/
`-- lvis/
    |-- valid/annotation/lvis_v1_val.json
    `-- valid/image/val2017/
```

The exact LVIS placement can be overridden with command-line arguments to the
real LVIS runner. Generated pipeline outputs are written under
`research/pipeline/outputs/` and are ignored by Git.

## Hosted assets

The coordinated release will publish:

- the portable AffordAny annotation dataset on Hugging Face and ModelScope;
- formal train/validation/test split manifests;
- a small example bundle for smoke tests;
- model-ready geometry and VLM caches where redistribution is permitted; and
- supervised and self-training checkpoints on the model hosting pages.

The hosted dataset must not contain absolute machine paths or symbolic links.
Every path in its manifests will be relative to the dataset root, and every
archive shard will include a SHA256 checksum.

The experimental pipeline completed 7,398 object workspaces. The public
benchmark must select the 5,334 objects with at least one validated part,
containing 10,633 part samples and 31,899 generated instruction pairs across
473 categories. The formal protocol uses 22,767 manifest rows after applying
its train/validation/test instruction policy.
