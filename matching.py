import cv2
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import differential_evolution

from transforms import apply_alignment, to_grayscale


def compute_correlation(img1: np.ndarray, img2: np.ndarray,
                        norm_method: str = "gradient",
                        corr_method: str = "pearson",
                        blur_sigma: float = 0.0) -> float:
    """Correlation over a fixed circular centre region.

    Expects grayscale (2D) input — callers must convert before passing.

    Two orthogonal knobs:

    `corr_method` selects how the two feature vectors are correlated:
      'pearson'  — linear correlation; assumes a linear intensity relationship.
      'spearman' — rank correlation (Pearson on the pixel ranks).  Invariant to
                   ANY monotonic intensity remap, so it is exactly gamma-
                   invariant.  Use when exposure/gamma differs between photos.
                   Slightly slower (an argsort per image) but very robust.

    `norm_method` selects the per-image feature compared:
      None        — raw grayscale, no normalization; correct only when the two
                    photos share the same exposure (e.g. gamma off).  The
                    cleanest baseline for geometry-only testing.
      'clahe'     — Contrast Limited Adaptive Histogram Equalization, applied
                    after the spatial transform.  Handles lighting/gamma
                    differences, BUT CLAHE is not rotation-equivariant: it
                    equalizes over an 8×8 tile grid that does not correspond
                    between an untransformed reference and a transformed match,
                    injecting angle-dependent distortion.  Use only when
                    lighting differs and rotation is small/known.
      'gradient'  — Sobel gradient magnitude.  Robust to additive brightness
                    and largely to gamma (an edge stays an edge), AND rotation-
                    equivariant (the magnitude field rotates with the image).
                    The recommended default for lighting-tolerant matching.

    `blur_sigma` > 0 applies a Gaussian blur before feature extraction,
    smoothing the correlation landscape (fewer spurious local maxima) at the
    cost of fine detail.  0 disables it.

    Common pipeline for every norm_method:
      1. Border mask built BEFORE any remap (warp fill is exactly 0; later
         steps can change those values, so the mask must use raw pixels).
      2. Fixed circular centre mask of radius 0.80 × min(w,h)/2 so the same
         pixel set is compared at every rotation angle — without it, angles
         near 0°/90° have larger valid areas than 45°, biasing the metric
         toward multiples of 90°.
      3. Border pixels filled with per-image median before any feature is
         computed.  For 'clahe' this removes the zero-spike that distorts tile
         histograms; for 'gradient' it removes the huge artificial edge the
         black border would create; for blur it stops black bleeding inward.
      4. Correlation (per corr_method) over the masked feature pixels.  Both
         Pearson and Spearman are invariant to global scale/offset, and the
         fixed mask keeps n constant across candidates, so no extra pixel-count
         normalization is needed.
    """
    _MIN_PIX   = 1000
    _CIRC_FRAC = 0.80   # circle radius as fraction of min(w,h)/2
    _CLIP      = 2.0
    _TILE      = (8, 8)

    h, w = img1.shape

    # Step 1 — border mask BEFORE any remap
    bord1 = img1 > 0
    bord2 = img2 > 0

    # Step 2 — fixed circular mask
    cy, cx = h / 2.0, w / 2.0
    r      = min(cx, cy) * _CIRC_FRAC
    ys, xs = np.ogrid[:h, :w]
    circle = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2

    valid = bord1 & bord2 & circle
    if valid.sum() < _MIN_PIX:
        return 0.0

    clahe = (cv2.createCLAHE(clipLimit=_CLIP, tileGridSize=_TILE)
             if norm_method == "clahe" else None)

    def _feature(gray: np.ndarray, bord: np.ndarray) -> np.ndarray:
        # Step 3 — fill borders with per-image median so they don't contaminate
        # blur / gradient / CLAHE statistics.
        med = int(np.median(gray[bord])) if bord.any() else 128
        filled = gray.copy()
        filled[~bord] = med

        if blur_sigma > 0:
            filled = cv2.GaussianBlur(filled, (0, 0), blur_sigma)

        if norm_method is None:
            return filled.astype(np.float64)
        if norm_method == "clahe":
            return clahe.apply(filled).astype(np.float64)
        if norm_method == "gradient":
            f = filled.astype(np.float32)
            gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
            return np.sqrt(gx * gx + gy * gy).astype(np.float64)
        raise ValueError(f"Unknown normalization method: {norm_method!r}")

    f1 = _feature(img1, bord1)
    f2 = _feature(img2, bord2)

    # Step 4 — correlate masked pixels
    a = f1[valid]
    b = f2[valid]
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    if corr_method == "pearson":
        r = pearsonr(a, b).statistic
    elif corr_method == "spearman":
        r = spearmanr(a, b).statistic
    else:
        raise ValueError(f"Unknown correlation method: {corr_method!r}")
    return 0.0 if np.isnan(r) else float(r)


def find_best_alignment(
    reference: np.ndarray,
    match: np.ndarray,
    # --- which transforms to search ---
    search_rotation: bool = True,
    search_zoom: bool = True,
    search_translation: bool = True,
    search_perspective: bool = True,
    # --- search ranges / step sizes ---
    step_rot: float = 1.0,
    step_persp: float = 5.0,
    persp_range: float = 30.0,
    step_trans: float = 5.0,
    trans_range: float = 50.0,
    zoom_values: tuple = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                          1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40),
    # --- correlation metric ---
    norm_method: str = "gradient",
    corr_method: str = "pearson",
    blur_sigma: float = 0.0,
    # --- iterative refinement ---
    n_passes: int = 3,
    verbose: bool = False,
) -> tuple[float, float, float, float, float, float, float, dict]:
    """Find the rotation, zoom, translation, and perspective to apply to match
    that maximizes correlation with reference.

    Each transform can be independently enabled or disabled via the search_*
    flags.  Disabled transforms keep their neutral value (angle=0, zoom=1,
    dx/dy=0, pan/tilt=0) and are held fixed during every search step.

    Search order within one pass: rotation → zoom → translation → perspective.
    Each step optimizes its own parameter(s) while holding ALL others at their
    current best value (not just the ones found earlier this pass).

    Iterative refinement (n_passes): the parameters are coupled — notably
    rotation and translation, because rotation is about the IMAGE CENTRE, so an
    uncorrected translation (off-centre subject) biases the rotation step that
    runs before it.  A single greedy sweep can therefore lock in a wrong angle.
    Repeating the sweep lets each parameter re-converge against the others'
    improved estimates (fixed-point iteration): pass 2's rotation step runs
    with pass 1's translation applied, so the subject is re-centred and the
    angle sharpens, and so on.  The loop stops early when a full pass changes
    nothing (the grid search has reached a fixed point).  Coupling requires at
    least two active searches, so a single active search runs only one pass.
    Iteration tightens coupling error; it cannot escape a wrong local maximum.

    For 1-D searches (rotation, zoom) the full correlation array is stored.
    For 2-D searches (translation, perspective) the full correlation matrix is
    stored and 1-D slices through the best point are extracted for plotting.
    The stored curves reflect the final (converged) pass.

    Returns (best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom,
             best_correlation, curves) where curves is a dict keyed by parameter
             name, each value being {'x', 'y', 'best', 'label'} arrays/scalars.
    """
    # CLAHE is applied inside compute_correlation after each transform, so
    # normalization is always in the aligned coordinate frame.  Pass raw
    # grayscale here; do NOT pre-apply CLAHE.
    ref_gray   = to_grayscale(reference)
    match_gray = to_grayscale(match)

    # Local shorthand so every candidate uses the same metric settings.
    def corr(ref, cand):
        return compute_correlation(ref, cand, norm_method=norm_method,
                                   corr_method=corr_method,
                                   blur_sigma=blur_sigma)

    best_angle = 0.0
    best_pan   = 0.0
    best_tilt  = 0.0
    best_dx    = 0.0
    best_dy    = 0.0
    best_zoom  = 1.0
    best_corr  = corr(ref_gray, match_gray)  # baseline

    curves: dict = {}

    # Coupling (esp. rotation↔translation) requires ≥2 active searches; with
    # fewer there is nothing to iterate, so collapse to a single pass.
    active = sum([search_rotation, search_zoom,
                  search_translation, search_perspective])
    passes = n_passes if active >= 2 else 1

    for p in range(passes):
        # Snapshot at the start of the pass to detect a fixed point afterwards.
        prev = (best_angle, best_zoom, best_dx, best_dy, best_pan, best_tilt)

        # Step 1: rotation — sweep angle, hold every other param at current best
        if search_rotation:
            rot_angles = np.arange(0.0, 360.0, step_rot)
            rot_corrs  = np.array([
                corr(ref_gray, apply_alignment(
                    match_gray, angle=float(a), zoom=best_zoom,
                    dx=best_dx, dy=best_dy, pan=best_pan, tilt=best_tilt))
                for a in rot_angles
            ])
            best_idx = int(np.argmax(rot_corrs))
            if rot_corrs[best_idx] > best_corr:
                best_corr  = float(rot_corrs[best_idx])
                best_angle = float(rot_angles[best_idx])
            curves['rotation'] = {'x': rot_angles, 'y': rot_corrs,
                                  'best': best_angle, 'label': 'Rotation (deg)'}

        # Step 2: zoom — sweep zoom, hold every other param at current best
        if search_zoom:
            zoom_arr   = np.array(zoom_values, dtype=float)
            zoom_corrs = np.array([
                corr(ref_gray, apply_alignment(
                    match_gray, angle=best_angle, zoom=float(z),
                    dx=best_dx, dy=best_dy, pan=best_pan, tilt=best_tilt))
                for z in zoom_arr
            ])
            best_idx = int(np.argmax(zoom_corrs))
            if zoom_corrs[best_idx] > best_corr:
                best_corr = float(zoom_corrs[best_idx])
                best_zoom = float(zoom_arr[best_idx])
            curves['zoom'] = {'x': zoom_arr, 'y': zoom_corrs,
                              'best': best_zoom, 'label': 'Zoom'}

        # Step 3: translation — sweep dx,dy; hold every other param at best
        if search_translation:
            dx_vals = np.arange(-trans_range, trans_range + step_trans, step_trans)
            dy_vals = np.arange(-trans_range, trans_range + step_trans, step_trans)
            trans_mat = np.zeros((len(dx_vals), len(dy_vals)))
            for i, dx in enumerate(dx_vals):
                for j, dy in enumerate(dy_vals):
                    trans_mat[i, j] = corr(
                        ref_gray, apply_alignment(
                            match_gray, angle=best_angle, zoom=best_zoom,
                            dx=float(dx), dy=float(dy),
                            pan=best_pan, tilt=best_tilt))
            best_ij = np.unravel_index(np.argmax(trans_mat), trans_mat.shape)
            if trans_mat[best_ij] > best_corr:
                best_corr = float(trans_mat[best_ij])
                best_dx   = float(dx_vals[best_ij[0]])
                best_dy   = float(dy_vals[best_ij[1]])
            bi = int(np.argmin(np.abs(dx_vals - best_dx)))
            bj = int(np.argmin(np.abs(dy_vals - best_dy)))
            curves['dx'] = {'x': dx_vals, 'y': trans_mat[:, bj],
                            'best': best_dx, 'label': 'X offset (px)'}
            curves['dy'] = {'x': dy_vals, 'y': trans_mat[bi, :],
                            'best': best_dy, 'label': 'Y offset (px)'}

        # Step 4: perspective — sweep pan,tilt; hold every other param at best
        if search_perspective:
            pan_vals  = np.arange(-persp_range, persp_range + step_persp, step_persp)
            tilt_vals = np.arange(-persp_range, persp_range + step_persp, step_persp)
            persp_mat = np.zeros((len(pan_vals), len(tilt_vals)))
            for i, pan in enumerate(pan_vals):
                for j, tilt in enumerate(tilt_vals):
                    persp_mat[i, j] = corr(
                        ref_gray, apply_alignment(
                            match_gray, angle=best_angle, zoom=best_zoom,
                            dx=best_dx, dy=best_dy,
                            pan=float(pan), tilt=float(tilt)))
            best_ij = np.unravel_index(np.argmax(persp_mat), persp_mat.shape)
            if persp_mat[best_ij] > best_corr:
                best_corr = float(persp_mat[best_ij])
                best_pan  = float(pan_vals[best_ij[0]])
                best_tilt = float(tilt_vals[best_ij[1]])
            bi = int(np.argmin(np.abs(pan_vals  - best_pan)))
            bj = int(np.argmin(np.abs(tilt_vals - best_tilt)))
            curves['pan']  = {'x': pan_vals,  'y': persp_mat[:, bj],
                              'best': best_pan,  'label': 'Pan (deg)'}
            curves['tilt'] = {'x': tilt_vals, 'y': persp_mat[bi, :],
                              'best': best_tilt, 'label': 'Tilt (deg)'}

        curr = (best_angle, best_zoom, best_dx, best_dy, best_pan, best_tilt)
        if verbose:
            print(f"    pass {p + 1}/{passes}: rot={best_angle:+.1f}"
                  f" zoom={best_zoom:.2f} dx={best_dx:+.0f} dy={best_dy:+.0f}"
                  f" pan={best_pan:+.1f} tilt={best_tilt:+.1f}"
                  f" corr={best_corr:.4f}")
        # Fixed point reached when a whole pass moved nothing (grid search snaps
        # to discrete values, so unchanged params compare exactly equal).
        if np.allclose(curr, prev, atol=1e-9):
            break

    return best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom, best_corr, curves


def find_best_alignment_de(
    reference: np.ndarray,
    match: np.ndarray,
    # --- which transforms to search ---
    search_rotation: bool = True,
    search_zoom: bool = True,
    search_translation: bool = True,
    search_perspective: bool = True,
    # --- search ranges (continuous bounds) ---
    persp_range: float = 30.0,
    trans_range: float = 50.0,
    zoom_values: tuple = (0.70, 1.40),
    # --- correlation metric (identical knobs to find_best_alignment) ---
    norm_method: str = "gradient",
    corr_method: str = "pearson",
    blur_sigma: float = 0.0,
    # --- differential-evolution controls ---
    popsize: int = 15,
    maxiter: int = 100,
    tol: float = 0.01,
    seed: int | None = None,
    downscale: float = 1.0,
    verbose: bool = False,
) -> tuple[float, float, float, float, float, float, float, dict]:
    """Global alignment search via Differential Evolution (an evolutionary /
    genetic-family optimizer).  Drop-in alternative to find_best_alignment with
    the SAME 8-tuple return signature.

    Unlike the greedy sweep, DE optimizes the full parameter vector JOINTLY over
    a population of candidates, so coupled parameters (notably rotation↔
    translation) are handled at once and the multimodal landscape is explored
    from many seeds — sidestepping the "wrong basin" failure of coordinate
    descent.  The objective is -correlation (DE minimizes); the correlation
    metric is reused unchanged via compute_correlation.

    Only enabled transforms enter the optimized vector; disabled ones are held
    at their neutral value (angle=0, zoom=1, dx/dy=0, pan/tilt=0).  Bounds:
    rotation [0,360), zoom [min,max](zoom_values), dx/dy ±trans_range,
    pan/tilt ±persp_range.

    downscale (0,1]: fraction to resize both images to BEFORE optimization.
    1.0 = off (full resolution, exact).  Lower values trade accuracy for speed
    since DE runs many evaluations.  Translation is in pixels, so dx/dy bounds
    are scaled by `downscale` and the recovered dx/dy are scaled back to full-
    resolution pixels; all other parameters are resolution-independent.  The
    reported correlation is always recomputed at full resolution for
    comparability with the greedy matcher.

    seed makes the (stochastic) search reproducible.  Returns curves with a
    single 'convergence' entry (best correlation vs DE generation) so the
    existing diagnostic plot panel still renders something meaningful.
    """
    ref_gray   = to_grayscale(reference)
    match_gray = to_grayscale(match)

    s = float(downscale)
    if s != 1.0:
        ref_opt   = cv2.resize(ref_gray,   None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        match_opt = cv2.resize(match_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        ref_opt, match_opt = ref_gray, match_gray

    def corr(ref, cand):
        return compute_correlation(ref, cand, norm_method=norm_method,
                                   corr_method=corr_method, blur_sigma=blur_sigma)

    zmin, zmax = float(min(zoom_values)), float(max(zoom_values))
    tb = trans_range * s   # translation bounds in optimization-space pixels

    # (name, low, high) for each enabled parameter, in vector order.
    specs = []
    if search_rotation:
        specs.append(('angle', 0.0, 360.0))
    if search_zoom:
        specs.append(('zoom', zmin, zmax))
    if search_translation:
        specs += [('dx', -tb, tb), ('dy', -tb, tb)]
    if search_perspective:
        specs += [('pan', -persp_range, persp_range),
                  ('tilt', -persp_range, persp_range)]

    neutral = dict(angle=0.0, zoom=1.0, dx=0.0, dy=0.0, pan=0.0, tilt=0.0)

    # Nothing enabled — return the neutral baseline.
    if not specs:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                corr(ref_gray, match_gray), {})

    names  = [n for n, _, _ in specs]
    bounds = [(lo, hi) for _, lo, hi in specs]

    def vec_to_params(x):
        p = dict(neutral)
        for n, v in zip(names, x):
            p[n] = float(v)
        return p

    def objective(x):
        p = vec_to_params(x)
        return -corr(ref_opt, apply_alignment(match_opt, **p))

    history: list[float] = []

    def cb(xk, convergence=None):
        history.append(-objective(xk))
        if verbose:
            print(f"    gen {len(history)}: corr={history[-1]:.4f}")

    result = differential_evolution(
        objective, bounds,
        popsize=popsize, maxiter=maxiter, tol=tol, seed=seed,
        init='latinhypercube', polish=True, callback=cb,
    )

    p = vec_to_params(result.x)
    # Convert translation back to full-resolution pixels (other params are
    # resolution-independent).
    best_angle = p['angle']
    best_zoom  = p['zoom']
    best_dx    = p['dx'] / s
    best_dy    = p['dy'] / s
    best_pan   = p['pan']
    best_tilt  = p['tilt']

    # Recompute correlation at full resolution for comparability.
    best_corr = corr(ref_gray, apply_alignment(
        match_gray, angle=best_angle, zoom=best_zoom,
        dx=best_dx, dy=best_dy, pan=best_pan, tilt=best_tilt))

    curves: dict = {}
    if history:
        y = np.array(history, dtype=float)
        curves['convergence'] = {
            'x': np.arange(1, len(y) + 1, dtype=float),
            'y': y,
            'best': float(int(np.argmax(y)) + 1),  # generation of best corr
            'label': 'DE generation',
        }

    return best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom, best_corr, curves
