import cv2
import numpy as np
import random
from pathlib import Path


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle (degrees, CCW in standard coords) around center.
    Output is the same size as input; corners outside the frame become black."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to a BGR or grayscale image via a lookup table.
    gamma < 1: brightens (simulates overexposure / bright environment)
    gamma > 1: darkens  (simulates underexposure / dim environment)
    """
    lut = np.array(
        [(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale. No-op if already 2D."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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
    """Pearson correlation coefficient between two same-size grayscale images.

    Returns a value in [-1, 1] where 1 = perfect pixel-by-pixel match.
    This is equivalent to OpenCV's TM_CCOEFF_NORMED (zero-mean NCC) applied
    to same-size images, but computed directly via np.corrcoef for clarity.
    Returns 0.0 if either image has no variance (e.g. all-black border result).
    """
    a = img1.ravel().astype(np.float64)
    b = img2.ravel().astype(np.float64)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def find_best_alignment(
    reference: np.ndarray,
    match: np.ndarray,
    step: float = 1.0,
) -> tuple[float, float]:
    """Find the rotation (degrees) to apply to match that maximizes correlation
    with reference.  Both images are preprocessed with CLAHE before comparison
    so that lighting differences do not interfere with the rotation search.
    Searches [0, 360) in increments of step degrees.

    Returns (best_angle, best_correlation).

    Note: if match = rotate(reference, theta), the best_angle should be
    approximately (360 - theta) % 360, i.e. the inverse rotation.
    """
    ref_clahe = apply_clahe(reference)
    match_clahe = apply_clahe(match)

    best_angle = 0.0
    best_corr = -np.inf

    for angle in np.arange(0.0, 360.0, step):
        rotated = rotate_image(match_clahe, angle)
        corr = compute_correlation(ref_clahe, rotated)
        if corr > best_corr:
            best_corr = corr
            best_angle = angle

    return best_angle, best_corr


def create_diagnostic_image(
    reference: np.ndarray,
    match_original: np.ndarray,
    match_aligned: np.ndarray,
    rotation_applied: float,
    gamma_applied: float,
    rotation_found: float,
    correlation: float,
    output_path: str,
) -> None:
    """Save a 3-panel diagnostic image:
      Left:   Reference image (the target)
      Center: Input image (simulated: reference + rotation + gamma)
      Right:  Best-aligned image (input rotated by rotation_found)
      Bottom: Metrics bar with applied gamma/rotation, found rotation, correlation
    """
    PANEL = 300
    GAP = 8
    LABEL_H = 32
    INFO_H = 78
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def prep(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (PANEL, PANEL), interpolation=cv2.INTER_AREA)

    def label_bar(text: str) -> np.ndarray:
        bar = np.full((LABEL_H, PANEL, 3), 30, dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
        cv2.putText(
            bar, text,
            ((PANEL - tw) // 2, (LABEL_H + th) // 2 - 2),
            FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )
        return bar

    panels = [prep(reference), prep(match_original), prep(match_aligned)]
    labels = ["Reference", "Input (simulated)", "Best Alignment"]
    cols = [np.vstack([p, label_bar(l)]) for p, l in zip(panels, labels)]

    gap = np.full((PANEL + LABEL_H, GAP, 3), 15, dtype=np.uint8)
    row = np.hstack([cols[0], gap, cols[1], gap, cols[2]])

    total_w = row.shape[1]
    sep = np.full((2, total_w, 3), 60, dtype=np.uint8)

    info = np.full((INFO_H, total_w, 3), 20, dtype=np.uint8)
    expected = (360.0 - rotation_applied) % 360.0
    lines = [
        f"Applied rotation: {rotation_applied:.1f} deg    "
        f"Applied gamma: {gamma_applied:.2f}    "
        f"Found rotation: {rotation_found:.1f} deg    "
        f"Expected: ~{expected:.1f} deg",
        f"Correlation score (CLAHE): {correlation:.4f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(info, line, (12, 26 + i * 30), FONT,
                    0.52, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, np.vstack([row, sep, info]))
    print(f"  Saved diagnostic: {output_path}")


def main():
    raw_dir = Path("raw_images")
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    image_files = sorted(
        list(raw_dir.glob("*.jpg"))
        + list(raw_dir.glob("*.jpeg"))
        + list(raw_dir.glob("*.png"))
    )
    if not image_files:
        print(f"No images found in {raw_dir}/")
        return

    for img_path in image_files:
        reference = cv2.imread(str(img_path))
        if reference is None:
            print(f"Could not load {img_path}, skipping.")
            continue

        # Simulate a second image: random rotation + random gamma shift.
        # In production these would be two separately captured photos of the same person.
        rotation_applied = random.uniform(10.0, 350.0)
        gamma_applied = random.choice([
            random.uniform(0.1, 0.5),   # overexposed / bright environment
            random.uniform(2.0, 5.0),   # underexposed / dim environment
        ])
        match = apply_gamma(rotate_image(
            reference, rotation_applied), gamma_applied)

        print(f"\n--- {img_path.name} ---")
        print(f"  Applied rotation : {rotation_applied:.1f} deg")
        print(f"  Applied gamma    : {gamma_applied:.2f}")
        print(f"  Searching 360 deg for best alignment (CLAHE + 1 deg steps)...")

        best_angle, best_corr = find_best_alignment(reference, match, step=1.0)
        expected = (360.0 - rotation_applied) % 360.0

        print(f"  Found rotation   : {best_angle:.1f} deg")
        print(f"  Expected ~        {expected:.1f} deg  (inverse of applied)")
        print(f"  Correlation score: {best_corr:.4f}")

        match_aligned = rotate_image(match, best_angle)

        create_diagnostic_image(
            reference=reference,
            match_original=match,
            match_aligned=match_aligned,
            rotation_applied=rotation_applied,
            gamma_applied=gamma_applied,
            rotation_found=best_angle,
            correlation=best_corr,
            output_path=str(out_dir / f"{img_path.stem}_diagnostic.png"),
        )


if __name__ == "__main__":
    main()
