# CLAUDE.md

Guidance for working in this project. Read alongside `README.md` (user-facing).

## Purpose

Find the geometric transform that aligns two images of the same subject (a
person, or *part* of a person — faces, limbs, ankles) captured from a different
viewpoint, angle, and lighting. On simulated data it measures how well the
matcher recovers a *known* applied transform.

Two entry points in `main.py`:
- `run_simulation_test()` — apply a random transform to each `raw_images/` image,
  run the matcher, save a diagnostic. This is what `python main.py` runs.
- `align_real_images(img1, img2, output_dir)` — match two real photos, no
  simulation. Returns a dict of found params + the aligned image.

## Architecture

| Module | Role |
|---|---|
| `transforms.py` | Pure image transforms; no matching logic. |
| `matching.py` | `compute_correlation` + two matchers. Depends on `transforms`. |
| `diagnostics.py` | Renders the diagnostic PNG. Depends only on cv2/numpy/matplotlib. |
| `main.py` | Orchestration + all configuration. Depends on all three. |

Data flow: `main` simulates a `match` image → matcher searches transform params
→ `apply_alignment` builds the aligned image → `create_diagnostic_image` saves it.

Stack: numpy, opencv-python-headless, scipy, matplotlib. Python ≥3.10, managed
with `uv`. Run with `uv run python main.py`; one-off checks with
`.venv/bin/python -c "..."`. No test suite — verify by running and inspecting
`output/*.png` (read the PNG to eyeball the table + panels).

## Core conventions

- **Images are BGR uint8** (OpenCV convention) or 2D grayscale. Transforms
  accept either where it makes sense; matching converts to grayscale first.
- **Off-frame fill is black (0)**, and `compute_correlation` relies on this:
  the border mask is `img > 0` computed *before* any feature remap.
- **Neutral parameter values** (a disabled transform collapses to these):
  `angle=0, pan=0, tilt=0, dx=0, dy=0, zoom=1`; lighting: `gamma=1, contrast=1,
  brightness=0, temp=0, shade=0`.
- **Canonical geometry composition** is `apply_alignment`, order
  `zoom → translate → perspective → rotate`. Used by both matchers and the
  output builder. Don't reorder without updating all three call sites.

### transforms.py notes

- `apply_perspective` builds the homography `H = K·R·K⁻¹` with `K` assuming the
  principal point at image center and focal length `f = w` (~53° FOV). The
  rotation `R` is built via `scipy.spatial.transform.Rotation.from_euler('xy',
  [tilt, pan], degrees=True)` — this is **exactly** the old hand-rolled
  `Ry(pan) @ Rx(tilt)` (verified bit-identical). It then prepends a translation
  so the warped center stays in frame.
- Photometric transforms (`apply_brightness_contrast`, `apply_color_temperature`,
  `apply_illumination_gradient`) are **simulation-only** — they are applied to
  build the synthetic second photo but are **never searched/recovered**. The
  correlation metric is chosen to be robust to them instead. `apply_color_temperature`
  is a no-op on grayscale.

### matching.py notes

- `compute_correlation` expects **grayscale** input. Pipeline: border mask (pre-remap)
  → fixed circular center mask (radius 0.80·min(w,h)/2, so the same pixel set is
  compared at every rotation angle — critical, removes 90°-multiple bias) → fill
  borders with per-image median → extract feature (`norm_method`) → correlate
  (`corr_method`).
- `norm_method`: `None` raw, `'gradient'` Sobel magnitude (lighting-robust AND
  rotation-equivariant — preferred for lighting tolerance), `'clahe'`
  (lighting-robust but NOT rotation-equivariant; injects angle-dependent error).
- `corr_method`: `'pearson'` linear, `'spearman'` rank-based (exactly gamma-invariant).
- Two matchers share the **same 8-tuple return signature**:
  `(angle, pan, tilt, dx, dy, zoom, correlation, curves)`.
  - `find_best_alignment_greedy`: coordinate descent, sweep order
    rotation→zoom→translation→perspective, each step holds all other params at
    current best. `n_passes` iterative refinement for coupled params
    (rotation↔translation), stops at a fixed point; collapses to 1 pass if <2
    searches active.
  - `find_best_alignment_de`: Differential Evolution, joint global optimization.
    `downscale` resizes before optimizing (translation bounds/results scaled
    accordingly; correlation always recomputed at full res). `curves` has a
    single `'convergence'` entry.
- `curves` is a dict keyed by param name (`rotation`/`zoom`/`dx`/`dy`/`pan`/`tilt`
  or `convergence`), each `{x, y, best, label}`. `main` adds an `expected` key
  (true inverse) for the diagnostic reference lines.

### diagnostics.py notes

- `create_diagnostic_image` renders 4 panels (Reference / Input / Best Alignment
  / Overlay) + a geometry table + a lighting footer + correlation-curve plots.
- **Table is geometry-only**: columns `ROT PAN TILT X Y ZOOM` (`N_COLS = 6`),
  rows `Applied / Found / Expected / Residual`, plus a `Step` row **only for the
  greedy matcher** (DE has no per-parameter step). `Expected` is the true inverse
  of `Applied`; `Residual = Found − Expected` (0 = perfect).
- **Lighting params go in a footer line** (`lighting_applied` dict with keys
  `gamma, contrast, brightness, temp, shade, shade_angle`), NOT in the table —
  they are applied but never recovered.
- `INFO_H` adapts: base `145` (greedy) / `128` (DE), `+17` for the lighting line.
  If you add/remove a table row or footer line, re-check this and the
  `ROW_Y0 / SEP2_Y / light_y / corr_y` arithmetic.

## Gotchas

- The greedy matcher function is `find_best_alignment_greedy` (there was an older
  `find_best_alignment` / `find_best_alignment_brute` naming that has been fully
  removed — don't reintroduce it).
- `matcher` strings are exactly `'greedy'` and `'de'`; `diagnostics` branches on
  `matcher == 'de'`.
- There are no CLI args — change behavior by editing the variables at the top of
  `run_simulation_test()`.

## Not yet implemented (intentional)

- **Subject masking / background exclusion.** The circular mask is geometric, so
  background still affects the score. Subjects can be any body part, so this
  needs a class-agnostic segmenter (SAM-family / `rembg`), not a person model.
  Integration: AND the foreground mask into `valid` in `compute_correlation`, and
  warp the match's mask with each candidate transform. Discussed but deferred.
