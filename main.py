import cv2
import random
from pathlib import Path

from transforms import (
    rotate_image, apply_gamma, apply_perspective, apply_translation, apply_zoom,
    apply_alignment,
)
from matching import find_best_alignment
from diagnostics import create_diagnostic_image


def run_simulation_test(
    raw_dir: str = "raw_images",
    output_dir: str = "output",
) -> None:
    """Run the full simulation pipeline on every image in raw_dir.

    For each image:
      1. Apply random rotation + perspective + translation + zoom + gamma
         to produce a simulated second photo.
      2. Run find_best_alignment to recover the transforms.
      3. Save a diagnostic PNG showing reference / input / aligned images
         alongside a table of applied vs found vs expected values.
    """
    # Which transforms to search — toggle to test subsets incrementally.
    # The simulation always applies all transforms regardless of these flags.
    search_rotation    = True
    search_zoom        = True
    search_translation = True
    search_perspective = True

    # Search step sizes — adjust to trade off speed vs precision.
    step_rot    = 1.0    # degrees, in-plane rotation
    step_persp  = 5.0    # degrees, pan and tilt
    persp_range = 30.0   # degrees, pan/tilt search range ±
    step_trans  = 5.0    # pixels, x and y translation
    trans_range = 50.0   # pixels, translation search range ±
    zoom_values = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                   1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40)
    zoom_step   = round(zoom_values[1] - zoom_values[0], 4)   # = 0.05

    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    image_files = sorted(
        list(raw_path.glob("*.jpg"))
        + list(raw_path.glob("*.jpeg"))
        + list(raw_path.glob("*.png"))
    )
    if not image_files:
        print(f"No images found in {raw_path}/")
        return

    for img_path in image_files:
        reference = cv2.imread(str(img_path))
        if reference is None:
            print(f"Could not load {img_path}, skipping.")
            continue

        # Simulate a second image: rotation + perspective + translation + zoom + gamma.
        # In production these would be two separately captured photos of the same person.
        rotation_applied = random.uniform(10.0, 350.0)
        pan_applied   = random.choice([random.uniform(-30, -5),  random.uniform(5, 30)])
        tilt_applied  = random.choice([random.uniform(-20, -5),  random.uniform(5, 20)])
        x_applied     = random.choice([random.uniform(-40, -10), random.uniform(10, 40)])
        y_applied     = random.choice([random.uniform(-40, -10), random.uniform(10, 40)])
        zoom_applied  = random.choice([random.uniform(0.75, 0.90), random.uniform(1.10, 1.35)])
        gamma_applied = random.choice([
            random.uniform(0.1, 0.5),   # overexposed / bright environment
            random.uniform(2.0, 5.0),   # underexposed / dim environment
        ])

        match = apply_gamma(
            apply_zoom(
                apply_translation(
                    apply_perspective(
                        rotate_image(reference, rotation_applied),
                        pan_applied, tilt_applied),
                    x_applied, y_applied),
                zoom_applied),
            gamma_applied,
        )

        print(f"\n--- {img_path.name} ---")
        print(f"  Applied  rot={rotation_applied:+.1f}  pan={pan_applied:+.1f}"
              f"  tilt={tilt_applied:+.1f}  x={x_applied:+.0f}"
              f"  y={y_applied:+.0f}  zoom={zoom_applied:.2f}  gamma={gamma_applied:.2f}")
        active = ", ".join(t for t, on in [
            ("rotation", search_rotation), ("zoom", search_zoom),
            ("translation", search_translation), ("perspective", search_perspective),
        ] if on) or "none"
        print(f"  Searching: {active}")

        best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom, best_corr = \
            find_best_alignment(
                reference, match,
                search_rotation=search_rotation,
                search_zoom=search_zoom,
                search_translation=search_translation,
                search_perspective=search_perspective,
                step_rot=step_rot,
                step_persp=step_persp,
                persp_range=persp_range,
                step_trans=step_trans,
                trans_range=trans_range,
                zoom_values=zoom_values,
            )

        exp_rot = (360.0 - rotation_applied) % 360.0
        print(f"  Found    rot={best_angle:+.1f}  pan={best_pan:+.1f}"
              f"  tilt={best_tilt:+.1f}  x={best_dx:+.0f}"
              f"  y={best_dy:+.0f}  zoom={best_zoom:.2f}")
        print(f"  Expected rot=~{exp_rot:.1f}  pan=~{-pan_applied:+.1f}"
              f"  tilt=~{-tilt_applied:+.1f}  x=~{-x_applied:+.0f}"
              f"  y=~{-y_applied:+.0f}  zoom=~{1/zoom_applied:.2f}")
        print(f"  Correlation: {best_corr:.4f}")

        match_aligned = apply_alignment(match, angle=best_angle, pan=best_pan,
                                        tilt=best_tilt, dx=best_dx, dy=best_dy,
                                        zoom=best_zoom)

        create_diagnostic_image(
            reference=reference,
            match_original=match,
            match_aligned=match_aligned,
            rotation_applied=rotation_applied,
            pan_applied=pan_applied,
            tilt_applied=tilt_applied,
            x_applied=x_applied,
            y_applied=y_applied,
            zoom_applied=zoom_applied,
            gamma_applied=gamma_applied,
            rotation_found=best_angle,
            pan_found=best_pan,
            tilt_found=best_tilt,
            x_found=best_dx,
            y_found=best_dy,
            zoom_found=best_zoom,
            correlation=best_corr,
            step_rot=step_rot,
            step_persp=step_persp,
            step_trans=step_trans,
            zoom_step=zoom_step,
            search_rotation=search_rotation,
            search_zoom=search_zoom,
            search_translation=search_translation,
            search_perspective=search_perspective,
            output_path=str(out_path / f"{img_path.stem}_diagnostic.png"),
        )


def align_real_images(
    img1_path: str,
    img2_path: str,
    output_dir: str | None = None,
) -> dict:
    """Find the best alignment between two real photos of the same subject.

    Unlike run_simulation_test, no transforms are applied — both images are
    taken as-is (e.g. two separate photos of the same person).  The function
    runs the same find_best_alignment search and returns the found parameters.

    Args:
        img1_path:  path to the reference image
        img2_path:  path to the image to align to the reference
        output_dir: if provided, saves the aligned image as aligned.png there

    Returns:
        dict with keys: rotation, pan, tilt, x_offset, y_offset, zoom,
                        correlation, aligned_image (np.ndarray)
    """
    reference = cv2.imread(img1_path)
    if reference is None:
        raise FileNotFoundError(f"Cannot load reference image: {img1_path}")
    match = cv2.imread(img2_path)
    if match is None:
        raise FileNotFoundError(f"Cannot load match image: {img2_path}")

    print(f"Aligning '{Path(img2_path).name}' to '{Path(img1_path).name}' ...")
    print(f"  Searching: rotation 360 deg | pan/tilt ±30 deg"
          f" | x/y ±50 px | zoom 0.70-1.40 ...")

    best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom, best_corr = \
        find_best_alignment(reference, match)

    match_aligned = apply_alignment(match, angle=best_angle, pan=best_pan,
                                    tilt=best_tilt, dx=best_dx, dy=best_dy,
                                    zoom=best_zoom)

    print(f"  Found    rot={best_angle:+.1f}  pan={best_pan:+.1f}"
          f"  tilt={best_tilt:+.1f}  x={best_dx:+.0f}"
          f"  y={best_dy:+.0f}  zoom={best_zoom:.2f}")
    print(f"  Correlation: {best_corr:.4f}")

    if output_dir:
        out_path = Path(output_dir) / "aligned.png"
        cv2.imwrite(str(out_path), match_aligned)
        print(f"  Saved: {out_path}")

    return {
        "rotation":      best_angle,
        "pan":           best_pan,
        "tilt":          best_tilt,
        "x_offset":      best_dx,
        "y_offset":      best_dy,
        "zoom":          best_zoom,
        "correlation":   best_corr,
        "aligned_image": match_aligned,
    }


if __name__ == "__main__":
    run_simulation_test()
