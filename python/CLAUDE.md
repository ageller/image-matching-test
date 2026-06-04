# CLAUDE.md

Guidance for working in this project. Read alongside `README.md` (user-facing).

## Purpose

Find the geometric transform that aligns two images of the same subject (a
person, or *part* of a person — faces, limbs, ankles) captured from a different
viewpoint, angle, and lighting. On simulated data it measures how well the
matcher recovers a *known* applied transform.

Two entry points in `main.py`:
- `run_simulation_test()` — apply a random transform to each `raw_images/` image,
  run the matcher, save a diagnostic. This is what `python main.py` runs. With
  `save_simulated=True` (default) it also writes `output/<stem>_simulated.png`
  per image — a synthetic second photo for testing `align_real_images`.
- `align_real_images(img1, img2, output_dir="output", matcher="de", ...)` — match
  two real photos, no simulation. Runs the same matcher/metric (knobs mirror
  `run_simulation_test`), saves `<stem2>_aligned.png` + a real-mode diagnostic
  `<stem2>_aligned_diagnostic.png`, and returns a dict of found params + the
  aligned image.

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
- **Two composed geometry warps**, exact functional inverses of each other,
  each applied as ONE `warpPerspective` (never as separate per-op warps —
  sequential warps clip to the frame at every step and compound interpolation
  blur, which can crop away enough content to make a transform unrecoverable):
  - `apply_capture_warp` — forward "capture" order `rotate → perspective →
    translate → zoom`. Used by the simulation to build the second photo.
  - `apply_alignment` — recovery order `zoom → translate → perspective →
    rotate`. Used by both matchers (to score candidates) and the output builder.
  Because the orders are reverses and the params negate/reciprocate,
  `apply_alignment(apply_capture_warp(x, p), inverse(p)) == x` up to off-frame
  clipping. Don't reorder either without updating the other and all call sites.

### transforms.py notes

- The geometry warps are assembled from 3×3 matrices via two private helpers:
  `_affine_to_3x3` (promote a cv2 2×3 affine) and `_perspective_matrix` (the
  pan/tilt homography). `_perspective_matrix` builds `H = K·R·K⁻¹` with `K`
  assuming the principal point at image center and `f = w` (~53° FOV); `R` is
  `scipy.spatial.transform.Rotation.from_euler('xy', [tilt, pan], degrees=True)`
  — **exactly** the old hand-rolled `Ry(pan) @ Rx(tilt)` (verified
  bit-identical). It prepends a translation so the warped center stays in frame.
  (There is no standalone `apply_perspective`/`apply_zoom`/`apply_translation`/
  `rotate_image` anymore — they were removed once both composed warps existed;
  don't reintroduce single-op warp helpers.)
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
  / Overlay) + a geometry table + (simulated only) a lighting footer +
  correlation-curve plots. Required args are the `*_found` values + `correlation`;
  the `*_applied` / `lighting_applied` args are **optional** and signal the mode.
- **Two modes**, detected by `rotation_applied is not None`:
  - **Simulated** (run_simulation_test passes ground truth): table rows
    `Applied / Found / Expected / Residual` + an applied-lighting footer.
    `Expected` is the true inverse of `Applied`; `Residual = Found − Expected`
    (0 = perfect).
  - **Real** (align_real_images omits ground truth): only the `Found` row, no
    lighting footer. `input_label` names the second photo.
- Table is geometry-only: columns `ROT PAN TILT X Y ZOOM` (`N_COLS = 6`). A
  greedy-only `Step` row is appended in both modes (DE has no per-parameter step).
- Lighting params (simulated mode) go in a footer line, NOT the table
  (`lighting_applied` dict: `gamma, contrast, brightness, temp, shade, shade_angle`)
  — applied but never recovered.
- `INFO_H` is **computed up front** from `n_rows` (4 simulated / 1 real, +1 for a
  greedy Step row) and whether the lighting line is present, via the
  `HDR_Y / SEP1_Y / ROW_Y0 / ROW_H / SEP2_Y / light_y / corr_y` chain near the top
  of the function. If you add/remove a row or footer line, adjust `n_rows` /
  that chain — don't hard-code heights.

## Known limitation: recoverability

Some inputs are **unrecoverable** — the matcher genuinely can't (and shouldn't be
expected to) align them, because the information is gone, not because of an
optimizer or metric bug:
- A strong **zoom-in** crops the subject off the fixed canvas; zooming back out
  only reveals black, not the lost content.
- **Severe lighting** (e.g. a heavy brightening gamma) flattens contrast until
  little structure remains.
When that happens the correct alignment has nothing to correlate, and the search
can settle on a different higher-correlation configuration. Verified for the old
img3 case: the geometrically-correct alignment scored *below* a spurious basin
under every `norm_method`, and high-pass normalization didn't fix it (it just
regressed the working cases) — so **don't try to "harden the metric" to recover
lost-information cases; it can't work.** The lever that helps is keeping inputs
recoverable. The simulation ranges are deliberately moderated for this:
**zoom-in capped at 1.20**, **gamma kept off the extremes** (bright floor 0.4,
dark ceiling 3.0). The same caveat applies to real photos. Note strong *combined*
distortion (large perspective + rotation together) can still leave a multimodal
landscape where the search lands in a wrong basin.

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
