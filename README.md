# Image Matching Test

Prototype for aligning two images of the same person (or part of a person)
taken from a different viewpoint, angle, and lighting. It searches for the
geometric transform that, when applied to one image, best correlates with the
other — and reports how well it recovered the known transform on simulated data.

## How it works

1. A reference image is loaded from `raw_images/`.
2. A simulated "second photo" is created by applying a random **geometric warp**
   (in-plane rotation → perspective pan/tilt → translation → zoom, composed into
   a single warp via `apply_capture_warp`) followed by random
   **photometric/lighting** effects (gamma, linear brightness/contrast,
   color-temperature shift, directional illumination gradient).
3. A matcher searches for the geometry that maximizes a correlation score
   between the reference and the transformed candidate. The lighting effects are
   *not* recovered — the correlation metric is chosen to be robust to them.
4. A diagnostic PNG is saved to `output/` showing reference / simulated input /
   best alignment / overlay, a table of applied-vs-found geometry, the applied
   lighting parameters, and per-parameter correlation curves.

In production the two images would be two separately captured photos; the
simulation exists to measure how well the matcher recovers a *known* transform.
See **Aligning two real images** below for the no-simulation path.

## Aligning two real images

`align_real_images(img1_path, img2_path, output_dir="output")` aligns a second
photo to a reference with no transforms applied — the real-world use case. It
runs the same matcher/metric (knobs mirror `run_simulation_test`, default
matcher `'de'`), saves `<stem2>_aligned.png` and a `<stem2>_aligned_diagnostic.png`,
and returns a dict of the found parameters plus the aligned image. Because there
is no ground-truth transform, the diagnostic shows only the **Found** row (no
Applied/Expected/Residual rows and no lighting footer).

To exercise it end-to-end, set `save_simulated = True` in `run_simulation_test`
(the default): each run writes `output/<stem>_simulated.png`, a synthetic
"second photo" you can align back to its original reference, e.g.

```python
align_real_images("raw_images/img1.jpg", "output/img1_simulated.png")
```

The `__main__` block at the bottom of `main.py` shows both entry points.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python main.py
```

Place test images (`.jpg`, `.jpeg`, or `.png`) in `raw_images/` before running.
Diagnostics are written to `output/<name>_diagnostic.png`.

All behavior is configured by editing the variables at the top of
`run_simulation_test()` in `main.py` (there are no CLI flags yet).

## Configuration (top of `run_simulation_test`)

**Which transforms to simulate** (`sim_*`) — build the synthetic second photo:
- `sim_rotation`, `sim_zoom`, `sim_translation`, `sim_perspective` — geometry
- `sim_gamma`, `sim_brightness_contrast`, `sim_color_temp`, `sim_shading` — lighting
- `save_simulated` — also write `output/<stem>_simulated.png` for each image (to
  feed into `align_real_images`)

**Which transforms to search** (`search_*`) — `rotation`, `zoom`, `translation`,
`perspective`. Independent of the `sim_*` flags; for a clean test the searched
set should match (or be a superset of) the applied set.

**Correlation metric:**
- `norm_method`: `None` (raw intensity), `'gradient'` (Sobel magnitude —
  lighting-robust and rotation-equivariant, recommended for lighting tolerance),
  or `'clahe'` (lighting-robust but *not* rotation-equivariant).
- `corr_method`: `'pearson'` (linear) or `'spearman'` (rank-based, exactly
  gamma-invariant — use when gamma/lighting differs).
- `blur_sigma`: >0 smooths the correlation landscape (fewer spurious local
  maxima); 0 disables.

**Matcher** (`matcher`):
- `'greedy'` — coordinate-descent sweep over a discrete grid, with iterative
  refinement (`n_passes`) so coupled parameters (esp. rotation↔translation)
  re-converge. Step sizes / ranges are set by `step_*` / `*_range` / `zoom_values`.
- `'de'` — Differential Evolution: global, *joint* optimization of all enabled
  parameters at once. Avoids the "wrong basin" failure of coordinate descent.
  Controlled by `de_popsize`, `de_maxiter`, `de_tol`, `de_seed`, `de_downscale`.

## Modules

| Module | Contents |
|---|---|
| `main.py` | `run_simulation_test()` (simulate + match + diagnose), `align_real_images()` (match two real photos). All knobs live here. |
| `transforms.py` | Geometric and photometric image transforms + `apply_alignment()` (the canonical composition). |
| `matching.py` | `compute_correlation()` and the two matchers. |
| `diagnostics.py` | `create_diagnostic_image()` and the correlation-curve plotter. |

### Key functions

| Function | Purpose |
|---|---|
| `apply_gamma(image, gamma)` | Nonlinear exposure — <1 brightens, >1 darkens |
| `apply_brightness_contrast(image, contrast, brightness)` | Linear exposure (contrast about mid-gray + offset) |
| `apply_color_temperature(image, temp)` | White-balance shift (>0 warm, <0 cool); no-op on grayscale |
| `apply_illumination_gradient(image, strength, angle)` | Directional brightness ramp (simulated side lighting) |
| `apply_capture_warp(image, angle, pan, tilt, dx, dy, zoom)` | Forward "capture" geometry (rotate→perspective→translate→zoom) as one composed warp — used to build the simulated photo |
| `apply_alignment(image, angle, pan, tilt, dx, dy, zoom)` | Recovery geometry (zoom→translate→perspective→rotate) as one composed warp; exact functional inverse of `apply_capture_warp` |
| `to_grayscale(image)` | BGR → grayscale |
| `compute_correlation(img1, img2, norm_method, corr_method, blur_sigma)` | Correlation over a fixed circular center region |
| `find_best_alignment_greedy(reference, match, ...)` | Coordinate-descent matcher |
| `find_best_alignment_de(reference, match, ...)` | Differential-Evolution matcher |
| `create_diagnostic_image(...)` | Save the 4-panel comparison + table + curves PNG. Simulated mode (ground truth supplied) shows Applied/Found/Expected/Residual + lighting footer; real-image mode (ground truth omitted) shows only the Found row |

## Known limitations / next steps

- **Background contributes to the score.** The correlation uses a fixed circular
  center mask, not a subject mask, so the background can influence the match. A
  future step is to segment the subject and exclude the background — and since
  subjects may be *any* body part (partial faces, limbs, ankles), this needs a
  class-agnostic segmenter (e.g. SAM-family or `rembg`) rather than a
  person-specific model.
- **Perspective is a planar warp.** The perspective homography models the image
  as a flat plane, so it only approximates small viewpoint changes; it cannot
  reproduce the parallax/self-occlusion of a real 3-D head seen from a very
  different angle.
- **Lighting is not relit in 3-D.** The illumination gradient reweights existing
  pixels; it does not generate new cast shadows or specular highlights.
- **Some inputs are unrecoverable — lost information can't be recovered.** A
  strong zoom-in crops the subject off the fixed canvas, and severe lighting
  (e.g. a heavy brightening gamma) can flatten contrast until little structure
  remains. When that happens the *correct* alignment has nothing left to match,
  and the matcher may settle on a different, higher-correlation configuration —
  no metric can recover detail that the input no longer contains. The
  simulation ranges are deliberately moderated (zoom-in capped at 1.20, gamma
  kept away from extremes) to keep generated cases recoverable, but **the same
  caveat applies to real photos**: a frame that is overly zoomed in, badly lit,
  or otherwise missing the subject's structure may not be alignable. Strong
  combined distortion (e.g. large perspective + rotation together) can also
  leave a multimodal landscape where the search lands in a wrong basin even when
  the input is otherwise fine.
