# Third-party dependencies

The full AffordAny data construction pipeline uses SAM 3 and SAM 3D Objects.
Their source code and checkpoints are not vendored into this release.

Place compatible checkouts at:

```text
third_party/
|-- sam3/
`-- sam-3d-objects/
```

Install and obtain checkpoints by following the upstream repositories. Review
their licenses and model terms before running or redistributing derived
assets. The lightweight pipeline and model checks do not require either
checkout; reconstruction, rendering, and segmentation do.

The pipeline discovers Conda environments named `sam3` and
`sam3d-objects`. Explicit environment and checkpoint paths can also be passed
through the stage configuration and command-line options.
