import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle (degrees, CCW in standard coords) around center.
    Output is the same size as input; corners outside the frame become black."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to a BGR or grayscale image via a lookup table.
    gamma < 1: brightens (simulates overexposure / bright environment)
    gamma > 1: darkens  (simulates underexposure / dim environment)
    """
    lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


def apply_perspective(image: np.ndarray, pan_deg: float, tilt_deg: float) -> np.ndarray:
    """Apply a 3D perspective warp to simulate a viewpoint change.

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
    """
    h, w = image.shape[:2]

    # --- Principal point assumed at image center; see docstring ---
    cx, cy = w / 2.0, h / 2.0
    f = float(w)  # focal length heuristic: image width ≈ 53 deg horizontal FOV

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0,  1]], dtype=np.float64)

    pan  = np.radians(pan_deg)
    tilt = np.radians(tilt_deg)

    Ry = np.array([[ np.cos(pan), 0, np.sin(pan)],   # rotation around Y (pan)
                   [ 0,           1, 0           ],
                   [-np.sin(pan), 0, np.cos(pan)]], dtype=np.float64)

    Rx = np.array([[1, 0,             0            ],  # rotation around X (tilt)
                   [0, np.cos(tilt), -np.sin(tilt) ],
                   [0, np.sin(tilt),  np.cos(tilt) ]], dtype=np.float64)

    R = Ry @ Rx  # pan first, then tilt
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
    H = T @ H

    return cv2.warpPerspective(image, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


def apply_translation(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate image by (dx, dy) pixels. dx>0 shifts right, dy>0 shifts down.
    Content shifted off-frame is filled with black."""
    h, w = image.shape[:2]
    M = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float64)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def apply_zoom(image: np.ndarray, scale: float) -> np.ndarray:
    """Zoom in (scale>1) or out (scale<1) around the image center.
    Content that falls outside the frame is filled with black."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, scale)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


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

    This is the canonical composition used everywhere: in the search
    (find_best_alignment), when constructing the aligned output image, and
    as the basis for testing different parameter combinations.  All parameters
    default to their neutral values so any subset can be applied.

    Order rationale:
      zoom first   — scale correction before any positional adjustment so that
                     dx/dy are measured in the final (scale-corrected) image space
      translate    — shift after scale so the offset isn't amplified by zoom
      perspective  — 3-D viewpoint warp applied in the image's natural orientation,
                     before any in-plane rotation changes that orientation
      rotate last  — in-plane tilt correction on top of everything else
    """
    img = apply_zoom(image, zoom)
    img = apply_translation(img, dx, dy)
    img = apply_perspective(img, pan, tilt)
    img = rotate_image(img, angle)
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale. No-op if already 2D."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
