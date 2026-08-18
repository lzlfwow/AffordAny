# AffordAny Project Page

Interactive project page for the AffordAny paper. It combines the figures used
in the paper with a browser-rendered 3D point-cloud viewer and a compact dataset
explorer.

## Local development

```bash
npm install
npm run dev
```

Build the static site with:

```bash
npm run build
```

The site is deployed from `.github/workflows/pages.yml`. Paths are relative so
the same build works locally and under the repository's GitHub Pages prefix.

## Preparing assets

The checked-in assets are web-ready derivatives of the paper figures and a
small set of release examples. Maintainers can regenerate them from the private
research workspace with:

```bash
python scripts/prepare_assets.py /path/to/openaffordance-workspace
```

The script downsamples point clouds and converts figures to WebP. It does not
copy datasets, checkpoints, or private experiment outputs into the release.
