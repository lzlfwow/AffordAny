# AffordAny release audit

This document records the boundary between the experimental workspace and the
public release candidate.

## Canonical public name

- Project: AffordAny
- Paper: AffordAny: VLM-Guided Open-World 3D Affordance Grounding from a
  Monocular RGB Image
- Initial release: v1.0.0

Internal implementation names ending in `_v5` and `_v1` are retained because
they are part of the tested Python import paths. They are presented publicly
as the AffordAny decoder and AffordAny self-training modules.

## Included

- M0-M11 dataset construction modules and the real LVIS runner.
- Shared pipeline utilities and per-stage documentation.
- The final supervised decoder source, training entry point, and evaluator.
- The final pseudo-label self-training source, manifest builder, and launch
  configuration.
- Portable repository path discovery and lightweight dependency definitions.

## Excluded

- Raw LVIS data and generated dataset outputs.
- Worker queues, SQLite state, logs, restart snapshots, and local symlinks.
- Feature caches, checkpoints, historical self-training rounds, and analysis
  artifacts.
- Paper figure-generation utilities and one-off experiment scripts.
- Vendored SAM 3 and SAM 3D Objects repositories and their checkpoints.
- Demo assets, which will be handled as a separate deployment artifact.

## Findings

1. The experimental workspace used several project names. The public name is
   now fixed as AffordAny.
2. The prior core-code bundle shortened two model directory names while the
   Python entry points expected their original names. The release candidate
   restores the expected directory names.
3. Historical launch scripts contain local storage paths. Only the portable
   final launch scripts are included.
4. The aggregated dataset index contains absolute worker paths, and its 7,398
   completed-object entries are absolute symbolic links. It must be
   materialized and rewritten before upload; the experimental directory
   cannot be published directly. The public benchmark is the validated subset
   of 5,334 objects, 10,633 parts, and 31,899 instruction pairs, not every
   completed worker object.
5. The code release uses Apache-2.0. Citation metadata is still pending.

## Decisions required before public release

- GitHub owner or organization and final repository slug.
- Dataset license and confirmation of which derived LVIS/image assets may be
  redistributed.
- Full author metadata for `CITATION.cff` and the arXiv submission.
- Hugging Face and ModelScope owner or organization names.
- Checkpoints are excluded from the initial public release.
- Hugging Face and ModelScope upload is deferred.

The strict release check remains intentionally blocked until citation metadata
is present.
