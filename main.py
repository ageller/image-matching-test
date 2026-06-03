import cv2
import random
from pathlib import Path

from transforms import (
    rotate_image, apply_gamma, apply_perspective, apply_translation, apply_zoom,
    apply_alignment, apply_brightness_contrast, apply_color_temperature,
    apply_illumination_gradient,
)
from matching import find_best_alignment_greedy, find_best_alignment_de
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
    # Which transforms to APPLY when simulating the second photo.
    # Toggle these to build a simpler test case (e.g. rotation only).
    # A disabled transform collapses to its neutral value (nothing applied).
    sim_rotation = True
    sim_zoom = True
    sim_translation = True
    sim_perspective = True
    # Photometric (lighting) effects — applied to the simulated second photo but
    # NOT recovered by the matcher (the gradient/rank correlation is designed to
    # be robust to them).  Each collapses to neutral when its flag is off.
    sim_gamma = True              # nonlinear exposure curve
    sim_brightness_contrast = True  # linear exposure (contrast + offset)
    sim_color_temp = True         # white-balance / color-temperature shift
    sim_shading = True            # directional illumination gradient

    # Which transforms to SEARCH during matching — toggle to test subsets.
    # Independent of the sim_* flags above, but for a clean test the searched
    # set should match (or be a superset of) the applied set.
    search_rotation = True
    search_zoom = True
    search_translation = True
    search_perspective = True

    # ================================================================
    # Matcher selection — picks WHICH search algorithm runs below.
    #   'greedy' — coordinate-descent sweep over a discrete grid.
    #   'de'     — Differential Evolution: global, joint optimization of all
    #              enabled parameters at once.
    # The settings below are split into three groups: shared (both matchers),
    # greedy-only, and de-only.  Settings outside the active matcher's group
    # are ignored.
    # ================================================================
    matcher = "de"

    # ---- SHARED settings (apply to BOTH matchers) ------------------
    # Correlation metric.
    #   norm_method (feature compared): None | 'clahe' | 'gradient'.
    #     None       — raw intensity, cleanest baseline.
    #     'gradient' — lighting-robust and rotation-equivariant.
    #     'clahe'    — lighting-robust but not rotation-equivariant.
    #   corr_method: 'pearson' | 'spearman'.
    #     'spearman' is rank-based, exactly gamma-invariant — use when gamma is on.
    #   blur_sigma > 0 smooths the correlation landscape (0 disables).
    norm_method = None
    corr_method = "spearman"
    blur_sigma = 0.0

    # Search ranges — greedy sweeps WITHIN these; DE uses them as bounds.
    persp_range = 30.0   # degrees, pan/tilt search range ±
    trans_range = 50.0   # pixels, x/y translation search range ±
    zoom_values = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                   1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40)
    # greedy treats this as the discrete grid of zoom values; DE uses only its
    # min and max as continuous bounds.

    # ---- GREEDY-only settings (ignored when matcher == 'de') -------
    # Step sizes — the grid resolution for each swept parameter (speed vs precision).
    step_rot = 1.0      # degrees, in-plane rotation
    step_persp = 5.0    # degrees, pan and tilt
    step_trans = 5.0    # pixels, x and y translation
    zoom_step = round(zoom_values[1] - zoom_values[0], 4)  # = 0.05; for diagnostics display
    # Iterative refinement: re-sweep up to n_passes times so coupled parameters
    # (esp. rotation↔translation) re-converge.  Stops early when a pass changes
    # nothing.  verbose prints each pass's parameter vector.
    n_passes = 3

    # ---- DE-only settings (ignored when matcher == 'greedy') -------
    de_popsize = 15     # population = popsize * n_enabled_params
    de_maxiter = 100    # max generations
    de_tol = 0.01       # convergence tolerance
    de_seed = 1234567   # reproducibility (DE is stochastic)
    de_downscale = 1.0  # 1.0 = full res; lower trades accuracy for speed

    random.seed(1234567)

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

        # Simulate a second image: geometric warp (rotation + perspective +
        # translation + zoom) followed by photometric/lighting effects.
        # In production these would be two separately captured photos of the same person.
        # Each transform is only applied if its sim_* flag is on; otherwise the
        # value collapses to neutral (rot=0, pan/tilt=0, dx/dy=0, zoom=1,
        # gamma=1, contrast=1, brightness=0, temp=0, shade=0).
        rotation_applied = random.uniform(10.0, 350.0) if sim_rotation else 0.0
        pan_applied = (random.choice([random.uniform(-30, -5), random.uniform(5, 30)])
                       if sim_perspective else 0.0)
        tilt_applied = (random.choice([random.uniform(-20, -5), random.uniform(5, 20)])
                        if sim_perspective else 0.0)
        x_applied = (random.choice([random.uniform(-40, -10), random.uniform(10, 40)])
                     if sim_translation else 0.0)
        y_applied = (random.choice([random.uniform(-40, -10), random.uniform(10, 40)])
                     if sim_translation else 0.0)
        zoom_applied = (random.choice([random.uniform(0.75, 0.90), random.uniform(1.10, 1.35)])
                        if sim_zoom else 1.0)
        gamma_applied = (random.choice([
            random.uniform(0.2, 0.5),   # overexposed / bright environment
            random.uniform(1.5, 4.0),   # underexposed / dim environment
        ]) if sim_gamma else 1.0)
        contrast_applied = (random.uniform(0.7, 1.3) if sim_brightness_contrast else 1.0)
        brightness_applied = (random.uniform(-40.0, 40.0) if sim_brightness_contrast else 0.0)
        temp_applied = (random.choice([random.uniform(-0.3, -0.1), random.uniform(0.1, 0.3)])
                        if sim_color_temp else 0.0)
        shade_applied = (random.uniform(0.2, 0.5) if sim_shading else 0.0)
        shade_angle_applied = (random.uniform(0.0, 360.0) if sim_shading else 0.0)

        # Geometric warp first, then photometric effects on top.
        match = rotate_image(reference, rotation_applied)
        match = apply_perspective(match, pan_applied, tilt_applied)
        match = apply_translation(match, x_applied, y_applied)
        match = apply_zoom(match, zoom_applied)
        match = apply_illumination_gradient(match, shade_applied, shade_angle_applied)
        match = apply_color_temperature(match, temp_applied)
        match = apply_gamma(match, gamma_applied)
        match = apply_brightness_contrast(match, contrast_applied, brightness_applied)

        lighting_applied = {
            'gamma': gamma_applied, 'contrast': contrast_applied,
            'brightness': brightness_applied, 'temp': temp_applied,
            'shade': shade_applied, 'shade_angle': shade_angle_applied,
        }

        print(f"\n--- {img_path.name} ---")
        print(f"  Applied  rot={rotation_applied:+.1f}  pan={pan_applied:+.1f}"
              f"  tilt={tilt_applied:+.1f}  x={x_applied:+.0f}"
              f"  y={y_applied:+.0f}  zoom={zoom_applied:.2f}")
        print(f"  Lighting gamma={gamma_applied:.2f}  contrast={contrast_applied:.2f}"
              f"  bright={brightness_applied:+.0f}  temp={temp_applied:+.2f}"
              f"  shade={shade_applied:.2f}@{shade_angle_applied:.0f}deg")
        active = ", ".join(t for t, on in [
            ("rotation", search_rotation), ("zoom", search_zoom),
            ("translation", search_translation), ("perspective", search_perspective),
        ] if on) or "none"
        print(f"  Searching: {active}")

        if matcher == "de":
            best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom, best_corr, curves = \
                find_best_alignment_de(
                    reference, match,
                    search_rotation=search_rotation,
                    search_zoom=search_zoom,
                    search_translation=search_translation,
                    search_perspective=search_perspective,
                    persp_range=persp_range,
                    trans_range=trans_range,
                    zoom_values=zoom_values,
                    norm_method=norm_method,
                    corr_method=corr_method,
                    blur_sigma=blur_sigma,
                    popsize=de_popsize,
                    maxiter=de_maxiter,
                    tol=de_tol,
                    seed=de_seed,
                    downscale=de_downscale,
                    verbose=True,
                )
        else:
            best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom, best_corr, curves = \
                find_best_alignment_greedy(
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
                    norm_method=norm_method,
                    corr_method=corr_method,
                    blur_sigma=blur_sigma,
                    n_passes=n_passes,
                    verbose=True,
                )

        exp_rot = (360.0 - rotation_applied) % 360.0

        # Annotate each curve with the expected (true inverse) value so the
        # diagnostic plots can draw a reference line alongside the found value.
        expected_by_key = {
            'rotation': exp_rot,
            'zoom':     1.0 / zoom_applied,
            'dx': -x_applied,
            'dy': -y_applied,
            'pan': -pan_applied,
            'tilt': -tilt_applied,
        }
        for key, exp_val in expected_by_key.items():
            if key in curves:
                curves[key]['expected'] = exp_val

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
            lighting_applied=lighting_applied,
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
            norm_method=norm_method,
            corr_method=corr_method,
            matcher=matcher,
            n_passes=n_passes,
            de_settings={
                'popsize': de_popsize, 'maxiter': de_maxiter, 'tol': de_tol,
                'seed': de_seed, 'downscale': de_downscale,
            },
            curves=curves,
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

    best_angle, best_pan, best_tilt, best_dx, best_dy, best_zoom, best_corr, _ = \
        find_best_alignment_greedy(reference, match)

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
