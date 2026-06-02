import cv2
import numpy as np

from transforms import apply_alignment, to_grayscale


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale and apply CLAHE (Contrast Limited Adaptive Histogram
    Equalization) to normalize local contrast.  Used as preprocessing before
    correlation so that lighting differences don't dominate the match score.
    clipLimit=2.0 and tileGridSize=(8,8) are standard defaults.
    """
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def compute_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Pearson correlation coefficient between two same-size grayscale images,
    computed only over pixels that are non-zero in both images.

    Zero pixels are border fill produced by warpAffine / warpPerspective and
    should not contribute to the match score — including them would dilute the
    signal with meaningless background agreement.

    Returns a value in [-1, 1] where 1 = perfect match, or 0.0 if there are
    fewer than 1000 valid pixels or either valid region has no variance.
    """
    mask = (img1 > 0) & (img2 > 0)
    if mask.sum() < 1000:
        return 0.0
    a = img1[mask].astype(np.float64)
    b = img2[mask].astype(np.float64)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


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
) -> tuple[float, float, float, float, float, float, float, dict]:
    """Find the rotation, zoom, translation, and perspective to apply to match
    that maximizes correlation with reference.  Both images are CLAHE-preprocessed.

    Each transform can be independently enabled or disabled via the search_*
    flags.  Disabled transforms return their neutral value (angle=0, zoom=1,
    dx/dy=0, pan/tilt=0) and are held fixed during every search step.

    Search order: rotation → zoom → translation → perspective.
    Each step holds all previously-found parameters fixed so later steps
    refine the residual rather than restart from scratch.

    For 1-D searches (rotation, zoom) the full correlation array is stored.
    For 2-D searches (translation, perspective) the full correlation matrix is
    stored and 1-D slices through the best point are extracted for plotting.

    Returns (best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom,
             best_correlation, curves) where curves is a dict keyed by parameter
             name, each value being {'x', 'y', 'best', 'label'} arrays/scalars.
    """
    ref_clahe   = apply_clahe(reference)
    match_clahe = apply_clahe(match)

    best_angle = 0.0
    best_pan   = 0.0
    best_tilt  = 0.0
    best_dx    = 0.0
    best_dy    = 0.0
    best_zoom  = 1.0
    best_corr  = compute_correlation(
        ref_clahe, apply_alignment(match_clahe))  # baseline at all-neutral

    curves: dict = {}

    # Step 1: rotation — collect full 1-D correlation array
    if search_rotation:
        rot_angles = np.arange(0.0, 360.0, step_rot)
        rot_corrs  = np.array([
            compute_correlation(ref_clahe,
                                apply_alignment(match_clahe, angle=float(a)))
            for a in rot_angles
        ])
        best_idx = int(np.argmax(rot_corrs))
        if rot_corrs[best_idx] > best_corr:
            best_corr  = float(rot_corrs[best_idx])
            best_angle = float(rot_angles[best_idx])
        curves['rotation'] = {'x': rot_angles, 'y': rot_corrs,
                              'best': best_angle, 'label': 'Rotation (deg)'}

    # Step 2: zoom — collect full 1-D correlation array
    if search_zoom:
        zoom_arr   = np.array(zoom_values, dtype=float)
        zoom_corrs = np.array([
            compute_correlation(ref_clahe,
                                apply_alignment(match_clahe, angle=best_angle,
                                                zoom=float(z)))
            for z in zoom_arr
        ])
        best_idx = int(np.argmax(zoom_corrs))
        if zoom_corrs[best_idx] > best_corr:
            best_corr = float(zoom_corrs[best_idx])
            best_zoom = float(zoom_arr[best_idx])
        curves['zoom'] = {'x': zoom_arr, 'y': zoom_corrs,
                          'best': best_zoom, 'label': 'Zoom'}

    # Step 3: translation — store full 2-D matrix, extract 1-D slices
    if search_translation:
        dx_vals = np.arange(-trans_range, trans_range + step_trans, step_trans)
        dy_vals = np.arange(-trans_range, trans_range + step_trans, step_trans)
        trans_mat = np.zeros((len(dx_vals), len(dy_vals)))
        for i, dx in enumerate(dx_vals):
            for j, dy in enumerate(dy_vals):
                trans_mat[i, j] = compute_correlation(
                    ref_clahe, apply_alignment(match_clahe, angle=best_angle,
                                               zoom=best_zoom,
                                               dx=float(dx), dy=float(dy)))
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

    # Step 4: perspective — store full 2-D matrix, extract 1-D slices
    if search_perspective:
        pan_vals  = np.arange(-persp_range, persp_range + step_persp, step_persp)
        tilt_vals = np.arange(-persp_range, persp_range + step_persp, step_persp)
        persp_mat = np.zeros((len(pan_vals), len(tilt_vals)))
        for i, pan in enumerate(pan_vals):
            for j, tilt in enumerate(tilt_vals):
                persp_mat[i, j] = compute_correlation(
                    ref_clahe, apply_alignment(match_clahe, angle=best_angle,
                                               zoom=best_zoom,
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

    return best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom, best_corr, curves
