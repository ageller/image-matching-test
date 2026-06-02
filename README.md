# Image Matching Test

Prototype for aligning two images of the same person taken at different orientations and lighting conditions.

## How it works

1. A reference image is loaded from `raw_images/`
2. A simulated second image is created by applying a random rotation and random gamma shift (to mimic different lighting)
3. Both images are preprocessed with CLAHE (contrast normalization) and a brute-force 1° rotation search finds the alignment that maximizes Pearson correlation
4. A diagnostic PNG is saved to `output/` showing the reference, simulated input, and best alignment side by side

## Setup

```bash
uv sync
```

## Run

```bash
uv run python main.py
```

Place test images (`.jpg` or `.png`) in `raw_images/` before running. Output diagnostics are written to `output/`.

## Key functions

| Function | Purpose |
|---|---|
| `rotate_image(image, angle)` | Rotate by degrees (CCW) around center |
| `apply_gamma(image, gamma)` | Gamma correction — <1 brightens, >1 darkens |
| `to_grayscale(image)` | BGR → grayscale |
| `apply_clahe(image)` | CLAHE contrast normalization (preprocessing for matching) |
| `compute_correlation(img1, img2)` | Pearson correlation coefficient [-1, 1] |
| `find_best_alignment(reference, match)` | CLAHE + brute-force rotation search |
| `create_diagnostic_image(...)` | Save 3-panel comparison PNG |
