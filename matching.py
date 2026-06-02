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
) -> tuple[float, float, float, float, float, float, float]:
    """Find the rotation, zoom, translation, and perspective to apply to match
    that maximizes correlation with reference.  Both images are CLAHE-preprocessed.

    Each transform can be independently enabled or disabled via the search_*
    flags.  Disabled transforms return their neutral value (angle=0, zoom=1,
    dx/dy=0, pan/tilt=0) and are held fixed during every search step.

    Search order: rotation → zoom → translation → perspective.
    Each step holds all previously-found parameters fixed so later steps
    refine the residual rather than restart from scratch.  apply_alignment
    is called with the full parameter set on every evaluation to keep the
    physical transform order (zoom→translate→perspective→rotate) consistent.

    Returns (best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom,
             best_correlation).
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

    if search_rotation:
        for angle in np.arange(0.0, 360.0, step_rot):
            corr = compute_correlation(
                ref_clahe, apply_alignment(match_clahe, angle=float(angle)))
            if corr > best_corr:
                best_corr  = corr
                best_angle = float(angle)

    if search_zoom:
        for z in zoom_values:
            corr = compute_correlation(
                ref_clahe, apply_alignment(match_clahe, angle=best_angle,
                                           zoom=float(z)))
            if corr > best_corr:
                best_corr = corr
                best_zoom = float(z)

    if search_translation:
        for dx in np.arange(-trans_range, trans_range + step_trans, step_trans):
            for dy in np.arange(-trans_range, trans_range + step_trans, step_trans):
                corr = compute_correlation(
                    ref_clahe, apply_alignment(match_clahe, angle=best_angle,
                                               zoom=best_zoom,
                                               dx=float(dx), dy=float(dy)))
                if corr > best_corr:
                    best_corr = corr
                    best_dx   = float(dx)
                    best_dy   = float(dy)

    if search_perspective:
        for pan in np.arange(-persp_range, persp_range + step_persp, step_persp):
            for tilt in np.arange(-persp_range, persp_range + step_persp, step_persp):
                corr = compute_correlation(
                    ref_clahe, apply_alignment(match_clahe, angle=best_angle,
                                               zoom=best_zoom,
                                               dx=best_dx, dy=best_dy,
                                               pan=float(pan), tilt=float(tilt)))
                if corr > best_corr:
                    best_corr = corr
                    best_pan  = float(pan)
                    best_tilt = float(tilt)

    return best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom, best_corr
