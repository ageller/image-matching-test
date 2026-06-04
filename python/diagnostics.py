import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def render_correlation_plots(curves: dict, total_w: int) -> np.ndarray:
    """Render per-parameter correlation curves as a matplotlib figure.

    Returns a BGR numpy array exactly total_w pixels wide.
    Curves are laid out in a 3-column grid, one subplot per parameter.
    Returns a zero-height array when curves is empty (no searches active).
    """
    if not curves:
        return np.zeros((0, total_w, 3), dtype=np.uint8)

    items    = list(curves.items())
    n_cols   = 3
    n_rows   = (len(items) + n_cols - 1) // n_cols
    DPI      = 96
    ROW_H_PX = 160

    fig = Figure(figsize=(total_w / DPI, n_rows * ROW_H_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor('#141414')

    for idx, (_, data) in enumerate(items):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.set_facecolor('#1c1c1c')

        x = np.asarray(data['x'], dtype=float)
        y = np.asarray(data['y'], dtype=float)

        ax.plot(x, y, color='#7ab8e8', linewidth=1.2)

        yrange = float(y.max() - y.min()) or 0.01

        # Found value — yellow dashed
        ax.axvline(data['best'], color='#f0c040', linewidth=1.2,
                   linestyle='--', alpha=0.9)
        ax.text(data['best'], float(y.max()) + yrange * 0.04,
                f'{data["best"]:.3g}',
                ha='center', va='bottom', color='#f0c040', fontsize=5.5)

        # Expected (true inverse) value — green dotted, if provided
        if 'expected' in data:
            ax.axvline(data['expected'], color='#88dd88', linewidth=1.0,
                       linestyle=':', alpha=0.85)
            ax.text(data['expected'], float(y.max()) + yrange * 0.04,
                    f'{data["expected"]:.3g}',
                    ha='center', va='bottom', color='#88dd88', fontsize=5.5)

        ax.set_xlabel(data['label'], color='#aaaaaa', fontsize=6.5, labelpad=2)
        ax.tick_params(colors='#888888', labelsize=6, length=2, width=0.5)
        for spine in ax.spines.values():
            spine.set_color('#444444')
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.grid(True, color='#2c2c2c', linewidth=0.5)

    fig.tight_layout(pad=0.5)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    bgr = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)

    if bgr.shape[1] != total_w:
        h = max(1, round(bgr.shape[0] * total_w / bgr.shape[1]))
        bgr = cv2.resize(bgr, (total_w, h))

    return bgr


def create_diagnostic_image(
    reference: np.ndarray,
    match_original: np.ndarray,
    match_aligned: np.ndarray,
    rotation_found: float,
    pan_found: float,
    tilt_found: float,
    x_found: float,
    y_found: float,
    zoom_found: float,
    correlation: float,
    # Ground-truth transform — supplied only by the simulation (run_simulation_test);
    # leave as None for real images (align_real_images), which have no ground truth.
    rotation_applied: float | None = None,
    pan_applied: float | None = None,
    tilt_applied: float | None = None,
    x_applied: float | None = None,
    y_applied: float | None = None,
    zoom_applied: float | None = None,
    lighting_applied: dict | None = None,
    input_label: str = "Input (simulated)",
    step_rot: float = 1.0,
    step_persp: float = 5.0,
    step_trans: float = 5.0,
    zoom_step: float = 0.05,
    search_rotation: bool = True,
    search_zoom: bool = True,
    search_translation: bool = True,
    search_perspective: bool = True,
    norm_method: str = "gradient",
    corr_method: str = "pearson",
    matcher: str = "greedy",
    n_passes: int = 1,
    de_settings: dict = None,
    curves: dict = None,
    output_path: str = "diagnostic.png",
) -> None:
    """Save a 4-panel diagnostic image with a table of recovered transforms.
      Left:   Reference image (the target)
      Center: Input image (the second photo being aligned)
      Right:  Best-aligned image (found transforms applied to input)
      Bottom: Table with columns ROT / PAN / TILT / X / Y / ZOOM (the geometry
              the matcher recovers) and rows:
                Applied — values used to create the simulated image *
                Found   — values recovered by the search algorithm
                Expected— the true inverse of Applied (perfect target) *
                Residual— Found minus Expected (0 = perfect recovery) *
                Step    — greedy search step per parameter (row omitted for DE,
                          which has no per-parameter step)
      Footer: the photometric/lighting parameters that were APPLIED but are not
              recovered (gamma/contrast/brightness/temp/shading) *, then a line
              with the correlation score and the matcher + its settings.

    * Rows/footer marked with an asterisk are ground-truth comparisons and are
    shown only in SIMULATED mode (run_simulation_test), detected by
    rotation_applied being supplied.  For REAL images (align_real_images) the
    ground-truth args are left None: only the Found row is shown, there is no
    lighting footer, and `input_label` should name the second photo.

    `lighting_applied` (simulated mode) is a dict with keys gamma, contrast,
    brightness, temp, shade, shade_angle (the lighting nuisance parameters).
    """
    PANEL   = 300
    GAP     = 8
    LABEL_H = 32
    FONT    = cv2.FONT_HERSHEY_SIMPLEX
    FS      = 0.42   # font scale for table text
    FT      = 1      # font thickness

    # Simulated mode (known ground truth) shows Applied/Found/Expected/Residual
    # rows + an applied-lighting footer.  Real-image mode has no ground truth, so
    # it shows only the Found row and no lighting footer.
    simulated = rotation_applied is not None

    # --- table/footer vertical layout (drives INFO_H, computed up front) ---
    HDR_Y  = 14            # header baseline
    SEP1_Y = HDR_Y + 6     # divider below header
    ROW_Y0 = SEP1_Y + 14   # baseline of first data row
    ROW_H  = 17
    # Data rows: simulated → Applied/Found/Expected/Residual (4); real → Found (1).
    # Plus a greedy-only Step row (DE has no per-parameter step).
    n_rows  = (4 if simulated else 1) + (0 if matcher == "de" else 1)
    SEP2_Y  = ROW_Y0 + n_rows * ROW_H + 2
    light_y = (SEP2_Y + ROW_H - 3) if simulated else None   # applied-lighting line
    corr_y  = (light_y + ROW_H) if simulated else (SEP2_Y + ROW_H - 3)
    INFO_H  = corr_y + 10

    # ------------------------------------------------------------------ panels
    def prep(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (PANEL, PANEL), interpolation=cv2.INTER_AREA)

    def label_bar(text: str) -> np.ndarray:
        bar = np.full((LABEL_H, PANEL, 3), 30, dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
        cv2.putText(bar, text, ((PANEL - tw) // 2, (LABEL_H + th) // 2 - 2),
                    FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        return bar

    ref_panel     = prep(reference)
    input_panel   = prep(match_original)
    aligned_panel = prep(match_aligned)
    # 4th panel: 50/50 alpha blend of reference and aligned image.
    # Misaligned features appear as ghosting; well-aligned areas look sharp.
    overlay_panel = cv2.addWeighted(ref_panel, 0.5, aligned_panel, 0.5, 0)

    panels_img = [ref_panel, input_panel, aligned_panel, overlay_panel]
    labels_img  = ["Reference", input_label, "Best Alignment", "Overlay"]
    cols_img = [np.vstack([p, label_bar(l)])
                for p, l in zip(panels_img, labels_img)]
    gap_strip = np.full((PANEL + LABEL_H, GAP, 3), 15, dtype=np.uint8)
    row = np.hstack([cols_img[0], gap_strip, cols_img[1],
                     gap_strip, cols_img[2], gap_strip, cols_img[3]])

    total_w = row.shape[1]   # 4*PANEL + 3*GAP = 1224 px for PANEL=300, GAP=8
    sep = np.full((2, total_w, 3), 60, dtype=np.uint8)
    info = np.full((INFO_H, total_w, 3), 20, dtype=np.uint8)

    # ------------------------------------------------------------------ table helpers
    DIV_COLOR  = (65, 65, 65)
    HDR_COLOR  = (150, 150, 150)
    DATA_COLOR = (200, 200, 200)
    CORR_COLOR = (230, 230, 230)

    def cell_w(text: str) -> int:
        return cv2.getTextSize(text, FONT, FS, FT)[0][0]

    def put_l(canvas, text, x, y, color=DATA_COLOR):
        cv2.putText(canvas, text, (x, y), FONT, FS, color, FT, cv2.LINE_AA)

    def put_r(canvas, text, right_x, y, color=DATA_COLOR):
        cv2.putText(canvas, text, (right_x - cell_w(text), y),
                    FONT, FS, color, FT, cv2.LINE_AA)

    def put_c(canvas, text, cx, y, color=HDR_COLOR):
        cv2.putText(canvas, text, (cx - cell_w(text) // 2, y),
                    FONT, FS, color, FT, cv2.LINE_AA)

    # Column right-edge x positions computed dynamically from total_w so the
    # table always fills the full image width regardless of panel count.
    LABEL_X   = 10
    LABEL_END = 90   # right edge of label column
    MARGIN    = 4    # gap between cell text and its right edge
    N_COLS    = 6
    _span     = total_w - LABEL_END
    COL_R     = [LABEL_END + round((i + 1) * _span / N_COLS) for i in range(N_COLS)]
    HEADERS   = ["ROT",  "PAN",  "TILT", "X",  "Y",  "ZOOM"]

    # ------------------------------------------------------------------ header row
    prev = LABEL_END
    for hdr, rx in zip(HEADERS, COL_R):
        put_c(info, hdr, (prev + rx) // 2, HDR_Y)
        prev = rx
    cv2.line(info, (0, SEP1_Y), (total_w, SEP1_Y), DIV_COLOR, 1)

    # ------------------------------------------------------------------ data rows
    # The Found row is always shown.  In simulated mode we also know the applied
    # transform, so we add Applied / Expected (its true inverse) / Residual.
    # (label, rot, pan, tilt, x, y, zoom)
    rows = [
        ("Found:",
         f"{rotation_found:+.1f}", f"{pan_found:+.1f}", f"{tilt_found:+.1f}",
         f"{x_found:+.0f}", f"{y_found:+.0f}", f"{zoom_found:.2f}"),
    ]
    if simulated:
        # Error = found − expected_inverse (0 means perfect recovery).
        # rotation: expected inverse = (360 − applied) % 360; normalize to [−180, 180].
        # pan/tilt/x/y: expected inverse = −applied, so error = found + applied.
        # zoom: expected inverse = 1 / applied, so error = found − 1/applied.
        exp_rot  = (360.0 - rotation_applied) % 360.0
        rot_err  = (rotation_found - exp_rot + 180) % 360 - 180
        rows = [
            ("Applied:",
             f"{rotation_applied:+.1f}", f"{pan_applied:+.1f}", f"{tilt_applied:+.1f}",
             f"{x_applied:+.0f}", f"{y_applied:+.0f}", f"{zoom_applied:.2f}"),
            rows[0],
            ("Expected:",
             f"{exp_rot:.1f}", f"{-pan_applied:+.1f}", f"{-tilt_applied:+.1f}",
             f"{-x_applied:+.0f}", f"{-y_applied:+.0f}", f"{1.0/zoom_applied:.2f}"),
            ("Residual:",
             f"{rot_err:+.1f}",
             f"{pan_found + pan_applied:+.1f}", f"{tilt_found + tilt_applied:+.1f}",
             f"{x_found + x_applied:+.0f}", f"{y_found + y_applied:+.0f}",
             f"{zoom_found - 1.0/zoom_applied:+.2f}"),
        ]

    # Step row: greedy shows each per-parameter step size ('---' when a
    # transform is disabled).  DE has no per-parameter step, so the row is
    # omitted entirely.
    if matcher != "de":
        rows.append(("Step:",
            f"{step_rot:.1f}"   if search_rotation    else "---",
            f"{step_persp:.1f}" if search_perspective else "---",
            f"{step_persp:.1f}" if search_perspective else "---",
            f"{step_trans:.1f}" if search_translation else "---",
            f"{step_trans:.1f}" if search_translation else "---",
            f"{zoom_step:.2f}"  if search_zoom        else "---",
        ))

    for i, (lbl, *vals) in enumerate(rows):
        y_pos = ROW_Y0 + i * ROW_H
        put_l(info, lbl, LABEL_X, y_pos)
        for val, rx in zip(vals, COL_R):
            if val:
                put_r(info, val, rx - MARGIN, y_pos)

    # ----------------------------------------------- separators, lighting & correlation
    # SEP2_Y / light_y / corr_y were computed up front (they drive INFO_H).
    cv2.line(info, (0, SEP2_Y), (total_w, SEP2_Y), DIV_COLOR, 1)

    # Lighting footer (simulated mode only): photometric nuisances that were
    # APPLIED but not recovered.
    if simulated:
        lg = lighting_applied or {}
        light_str = (f"Lighting applied:  gamma={lg.get('gamma', 1.0):.2f}"
                     f"   contrast={lg.get('contrast', 1.0):.2f}"
                     f"   bright={lg.get('brightness', 0.0):+.0f}"
                     f"   temp={lg.get('temp', 0.0):+.2f}"
                     f"   shade={lg.get('shade', 0.0):.2f}@{lg.get('shade_angle', 0.0):.0f}deg")
        put_l(info, light_str, LABEL_X, light_y, HDR_COLOR)

    put_l(info, f"Correlation score ({norm_method or 'none'}/{corr_method}): {correlation:.4f}",
          LABEL_X, corr_y, CORR_COLOR)

    # Matcher + its settings, right-aligned on the same footer line.
    if matcher == "de":
        s = de_settings or {}
        matcher_str = (f"Matcher: DE  pop={s.get('popsize')}  iter={s.get('maxiter')}"
                       f"  tol={s.get('tol')}  seed={s.get('seed')}"
                       f"  scale={s.get('downscale')}")
    else:
        matcher_str = f"Matcher: greedy  n_passes={n_passes}"
    put_r(info, matcher_str, total_w - LABEL_X, corr_y, CORR_COLOR)

    # ------------------------------------------------------------------ vertical dividers
    # Draw between label column and first data column, and between each data column pair.
    # Lines span from just below the header separator to just above the correlation separator.
    for x in [LABEL_END] + COL_R[:-1]:
        cv2.line(info, (x, SEP1_Y + 1), (x, SEP2_Y - 1), DIV_COLOR, 1)

    parts = [row, sep, info]
    plot_strip = render_correlation_plots(curves or {}, total_w)
    if plot_strip.shape[0] > 0:
        parts += [np.full((2, total_w, 3), 40, dtype=np.uint8), plot_strip]

    cv2.imwrite(output_path, np.vstack(parts))
    print(f"  Saved diagnostic: {output_path}")
