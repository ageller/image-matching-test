import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to a BGR or grayscale image via a lookup table.
    gamma < 1: brightens (simulates overexposure / bright environment)
    gamma > 1: darkens  (simulates underexposure / dim environment)
    """
    lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


def apply_brightness_contrast(image: np.ndarray, contrast: float = 1.0,
                              brightness: float = 0.0) -> np.ndarray:
    """Linear exposure change: out = contrast · (image − 128) + 128 + brightness.

    Models the linear part of an exposure difference between two captures
    (gamma covers the nonlinear curve).  Contrast pivots about mid-gray (128)
    so it scales dynamic range without shifting the midtone; brightness then
    adds a flat offset.
      contrast   > 1 increases contrast, < 1 flattens it
      brightness > 0 brightens, < 0 darkens (in 0-255 units)
    """
    out = contrast * (image.astype(np.float64) - 128.0) + 128.0 + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_color_temperature(image: np.ndarray, temp: float) -> np.ndarray:
    """Shift white balance / color temperature of a BGR image.
    temp > 0 warms (boosts red, cuts blue); temp < 0 cools (boosts blue, cuts
    red).  Simulates different ambient lighting between two photos.  No-op on a
    grayscale (2D) image, which has no color channels to shift.
    """
    if image.ndim == 2:
        return image
    out = image.astype(np.float64)
    out[..., 2] *= (1.0 + temp)   # R (BGR order: channel 2)
    out[..., 0] *= (1.0 - temp)   # B (BGR order: channel 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_illumination_gradient(image: np.ndarray, strength: float,
                                angle_deg: float) -> np.ndarray:
    """Multiply the image by a smooth linear brightness ramp to simulate
    directional lighting (light coming from a different side in the second
    photo).  The ramp runs along `angle_deg` (0 = left→right, 90 = top→bottom)
    and scales pixels from (1 − strength) on the dark side to (1 + strength) on
    the bright side.  strength = 0 is a no-op.  Works for grayscale and BGR.

    NOTE: this only reweights existing pixel intensities; it does not relight
    the 3-D surface, so it cannot reproduce cast shadows or specular highlights
    that move with the light source.
    """
    h, w = image.shape[:2]
    ang = np.radians(angle_deg)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    # Normalized so image center is 0 and edges are ±1 along each axis.
    xn = (xs - w / 2.0) / (w / 2.0)
    yn = (ys - h / 2.0) / (h / 2.0)
    proj = np.clip(np.cos(ang) * xn + np.sin(ang) * yn, -1.0, 1.0)
    factor = 1.0 + strength * proj
    out = image.astype(np.float64)
    if out.ndim == 3:
        factor = factor[..., None]
    return np.clip(out * factor, 0, 255).astype(np.uint8)


def _affine_to_3x3(affine_2x3: np.ndarray) -> np.ndarray:
    """Promote a 2×3 affine matrix (cv2 convention) to a 3×3 homography."""
    return np.vstack([affine_2x3, [0.0, 0.0, 1.0]])


def _perspective_matrix(w: int, h: int, pan_deg: float, tilt_deg: float) -> np.ndarray:
    """Build the 3×3 perspective homography for a viewpoint change (pan/tilt).

    Models the image as a flat plane viewed through a pinhole camera, then
    rotates the plane around the Y-axis (pan left/right) and X-axis (tilt
    up/down).  The warp is the homography H = K · R · K⁻¹, where:
      K   = camera intrinsic matrix (focal length + principal point)
      R   = combined 3D rotation matrix (pan then tilt)
      K⁻¹ = inverse intrinsics

    ASSUMPTION: principal point (optical axis intercept) is fixed at image
    center (cx = w/2, cy = h/2).  This is a standard default and is
    reasonable for portrait photos where the subject is roughly centered.
    If the subject is significantly off-center, or if the true camera
    intrinsics are known, cx/cy should be updated here.

    Focal length is approximated as image width (f = w), corresponding to
    a ~53 deg horizontal FOV.  Adjust if the actual FOV is known.

    pan_deg  > 0: subject appears to turn right (camera shifts left)
    tilt_deg > 0: subject appears to tilt upward (camera shifts down)

    Returns the 3×3 matrix (not the warped image) so apply_alignment and
    apply_capture_warp can compose it with the other transforms into one warp.
    """
    # --- Principal point assumed at image center; see docstring ---
    cx, cy = w / 2.0, h / 2.0
    f = float(w)  # focal length heuristic: image width ≈ 53 deg horizontal FOV

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0,  1]], dtype=np.float64)

    # Extrinsic 'xy' sequence (fixed world axes): rotate about X by tilt, then
    # about Y by pan.  scipy composes this as Ry(pan) @ Rx(tilt), matching the
    # previous hand-built `R = Ry @ Rx`.  Using scipy avoids hand-rolling the
    # rotation matrices and their composition.
    R = Rotation.from_euler('xy', [tilt_deg, pan_deg], degrees=True).as_matrix()
    H = K @ R @ np.linalg.inv(K)

    # The perspective warp shifts the image center: under a pure pan θ the
    # center maps to (cx + f·tan(θ), cy), pushing content off-frame.  Compute
    # where the center lands and prepend a translation to bring it back.
    # This keeps the subject (assumed near image center) visible after warping.
    p = H @ np.array([cx, cy, 1.0])
    p /= p[2]
    T = np.eye(3, dtype=np.float64)
    T[0, 2] = cx - p[0]
    T[1, 2] = cy - p[1]
    return T @ H


def apply_alignment(
    image: np.ndarray,
    angle: float = 0.0,
    pan: float = 0.0,
    tilt: float = 0.0,
    dx: float = 0.0,
    dy: float = 0.0,
    zoom: float = 1.0,
) -> np.ndarray:
    """Apply all alignment transforms in the physically correct order:
      zoom → translate → perspective → rotate

    This is the canonical composition used everywhere: inside the matchers
    (find_best_alignment_greedy / _de) to score candidates, when constructing
    the aligned output image, and as the basis for testing different parameter
    combinations.  All parameters default to their neutral values so any subset
    can be applied.  It is the exact functional inverse of apply_capture_warp.

    Order rationale:
      zoom first   — scale correction before any positional adjustment so that
                     dx/dy are measured in the final (scale-corrected) image space
      translate    — shift after scale so the offset isn't amplified by zoom
      perspective  — 3-D viewpoint warp applied in the image's natural orientation,
                     before any in-plane rotation changes that orientation
      rotate last  — in-plane tilt correction on top of everything else

    The four steps are composed into ONE 3×3 homography and applied with a
    single warpPerspective.  This matters: applying them as four separate warps
    clips to the frame at every intermediate step, so content pushed off-frame
    by (say) the zoom is lost to black even if a later step would bring it back.
    Composing first means only the FINAL position clips, and the image is
    resampled once instead of four times (less interpolation blur).
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    H_zoom  = _affine_to_3x3(cv2.getRotationMatrix2D((cx, cy), 0.0, zoom))
    H_trans = np.array([[1.0, 0.0, dx],
                        [0.0, 1.0, dy],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
    H_persp = _perspective_matrix(w, h, pan, tilt)
    H_rot   = _affine_to_3x3(cv2.getRotationMatrix2D((cx, cy), angle, 1.0))

    # Composition order is the reverse of application order: the matrix applied
    # last (rotation) sits leftmost.  H · x first zooms x, then translates, …
    H = H_rot @ H_persp @ H_trans @ H_zoom

    return cv2.warpPerspective(image, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


def apply_capture_warp(
    image: np.ndarray,
    angle: float = 0.0,
    pan: float = 0.0,
    tilt: float = 0.0,
    dx: float = 0.0,
    dy: float = 0.0,
    zoom: float = 1.0,
) -> np.ndarray:
    """Forward 'capture' geometry as a SINGLE composed warp, in the order
      rotate → perspective → translate → zoom

    Models how a second photo is framed (subject orientation, then viewpoint,
    then framing offset, then focal-length/zoom).  This is the exact functional
    inverse of apply_alignment — same operations, reverse order — so
    apply_alignment(apply_capture_warp(x, p), inverse(p)) recovers x up to
    off-frame clipping.

    Used by the simulation to build the synthetic second photo.  As with
    apply_alignment, composing into one warp (vs four sequential ones) avoids
    compounding interpolation blur and intermediate-frame clipping — the latter
    can otherwise crop away so much content that the transform is no longer
    recoverable.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    H_rot   = _affine_to_3x3(cv2.getRotationMatrix2D((cx, cy), angle, 1.0))
    H_persp = _perspective_matrix(w, h, pan, tilt)
    H_trans = np.array([[1.0, 0.0, dx],
                        [0.0, 1.0, dy],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
    H_zoom  = _affine_to_3x3(cv2.getRotationMatrix2D((cx, cy), 0.0, zoom))

    # rotate applied first (rightmost), zoom last (leftmost).
    H = H_zoom @ H_trans @ H_persp @ H_rot

    return cv2.warpPerspective(image, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale. No-op if already 2D."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
