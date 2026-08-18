# Dataset Construction Pipeline

This directory is the execution root for the released dataset construction
pipeline. Each stage has its own module folder, shared code lives in `common/`,
and `module_real_lvis_runner/` provides the end-to-end LVIS entry point.

Generated datasets are written under `outputs/` at runtime and are not part of
this source release.
