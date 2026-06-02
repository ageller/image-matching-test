import cv2
import numpy as np


def create_diagnostic_image(
    reference: np.ndarray,
    match_original: np.ndarray,
    match_aligned: np.ndarray,
    rotation_applied: float,
    pan_applied: float,
    tilt_applied: float,
    x_applied: float,
    y_applied: float,
    zoom_applied: float,
    gamma_applied: float,
    rotation_found: float,
    pan_found: float,
    tilt_found: float,
    x_found: float,
    y_found: float,
    zoom_found: float,
    correlation: float,
    step_rot: float,
    step_persp: float,
    step_trans: float,
    zoom_step: float,
    search_rotation: bool = True,
    search_zoom: bool = True,
    search_translation: bool = True,
    search_perspective: bool = True,
    output_path: str = "diagnostic.png",
) -> None:
    """Save a 3-panel diagnostic image with a right-aligned table of transforms.
      Left:   Reference image (the target)
      Center: Input image (simulated: rotation + perspective + translate + zoom + gamma)
      Right:  Best-aligned image (found transforms applied to input)
      Bottom: Table with columns ROT / PAN / TILT / X / Y / ZOOM / GAMMA and rows:
                Applied — values used to create the simulated image
                Found   — values recovered by the search algorithm
                Error   — found minus expected-inverse (0 = perfect recovery)
                Step    — search step size used for each parameter
    """
    PANEL   = 300
    GAP     = 8
    LABEL_H = 32
    INFO_H  = 145
    FONT    = cv2.FONT_HERSHEY_SIMPLEX
    FS      = 0.42   # font scale for table text
    FT      = 1      # font thickness

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

    panels_img = [prep(reference), prep(match_original), prep(match_aligned)]
    labels_img  = ["Reference", "Input (simulated)", "Best Alignment"]
    cols_img = [np.vstack([p, label_bar(l)])
                for p, l in zip(panels_img, labels_img)]
    gap_strip = np.full((PANEL + LABEL_H, GAP, 3), 15, dtype=np.uint8)
    row = np.hstack([cols_img[0], gap_strip, cols_img[1], gap_strip, cols_img[2]])

    total_w = row.shape[1]   # 916 px for PANEL=300, GAP=8
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

    # Column right-edge x positions; values are right-aligned to (rx - MARGIN).
    # 7 data columns fill total_w=916; label column occupies x < LABEL_END.
    LABEL_X   = 10
    LABEL_END = 90   # right edge of label column
    MARGIN    = 4    # gap between text and column right edge
    COL_R     = [195, 313, 431, 544, 657, 770, 883]
    HEADERS   = ["ROT",  "PAN",  "TILT", "X",  "Y",  "ZOOM",  "GAMMA"]

    # ------------------------------------------------------------------ header row
    HDR_Y  = 14
    SEP1_Y = HDR_Y + 6   # horizontal line below header
    prev = LABEL_END
    for hdr, rx in zip(HEADERS, COL_R):
        put_c(info, hdr, (prev + rx) // 2, HDR_Y)
        prev = rx
    cv2.line(info, (0, SEP1_Y), (total_w, SEP1_Y), DIV_COLOR, 1)

    # ------------------------------------------------------------------ data rows
    # Error = found − expected_inverse (0 means perfect recovery).
    # For rotation: expected inverse = (360 − applied) % 360; normalise to [−180, 180].
    # For pan/tilt/x/y: expected inverse = −applied, so error = found + applied.
    # For zoom: expected inverse = 1 / applied, so error = found − 1/applied.
    exp_rot  = (360.0 - rotation_applied) % 360.0
    rot_err  = rotation_found - exp_rot
    rot_err  = (rot_err + 180) % 360 - 180   # normalise to [−180, 180]
    pan_err  = pan_found  + pan_applied
    tilt_err = tilt_found + tilt_applied
    x_err    = x_found    + x_applied
    y_err    = y_found    + y_applied
    zoom_err = zoom_found - (1.0 / zoom_applied)

    ROW_Y0 = SEP1_Y + 14   # baseline of first data row
    ROW_H  = 17

    # (label, rot, pan, tilt, x, y, zoom, gamma)
    rows = [
        ("Applied:",
         f"{rotation_applied:+.1f}", f"{pan_applied:+.1f}", f"{tilt_applied:+.1f}",
         f"{x_applied:+.0f}", f"{y_applied:+.0f}",
         f"{zoom_applied:.2f}", f"{gamma_applied:.2f}"),
        ("Found:",
         f"{rotation_found:+.1f}", f"{pan_found:+.1f}", f"{tilt_found:+.1f}",
         f"{x_found:+.0f}", f"{y_found:+.0f}",
         f"{zoom_found:.2f}", ""),
        ("Expected:",
         f"{exp_rot:.1f}", f"{-pan_applied:+.1f}", f"{-tilt_applied:+.1f}",
         f"{-x_applied:+.0f}", f"{-y_applied:+.0f}",
         f"{1.0/zoom_applied:.2f}", ""),
        ("Residual:",
         f"{rot_err:+.1f}", f"{pan_err:+.1f}", f"{tilt_err:+.1f}",
         f"{x_err:+.0f}", f"{y_err:+.0f}",
         f"{zoom_err:+.2f}", ""),
        ("Step:",
         f"{step_rot:.1f}"   if search_rotation    else "---",
         f"{step_persp:.1f}" if search_perspective  else "---",
         f"{step_persp:.1f}" if search_perspective  else "---",
         f"{step_trans:.1f}" if search_translation  else "---",
         f"{step_trans:.1f}" if search_translation  else "---",
         f"{zoom_step:.2f}"  if search_zoom         else "---",
         ""),
    ]

    for i, (lbl, *vals) in enumerate(rows):
        y_pos = ROW_Y0 + i * ROW_H
        put_l(info, lbl, LABEL_X, y_pos)
        for val, rx in zip(vals, COL_R):
            if val:
                put_r(info, val, rx - MARGIN, y_pos)

    # ------------------------------------------------------------------ separators & correlation
    SEP2_Y = ROW_Y0 + len(rows) * ROW_H + 2
    cv2.line(info, (0, SEP2_Y), (total_w, SEP2_Y), DIV_COLOR, 1)
    put_l(info, f"Correlation score (CLAHE): {correlation:.4f}",
          LABEL_X, SEP2_Y + ROW_H - 2, CORR_COLOR)

    # ------------------------------------------------------------------ vertical dividers
    # Draw between label column and first data column, and between each data column pair.
    # Lines span from just below the header separator to just above the correlation separator.
    for x in [LABEL_END] + COL_R[:-1]:
        cv2.line(info, (x, SEP1_Y + 1), (x, SEP2_Y - 1), DIV_COLOR, 1)

    cv2.imwrite(output_path, np.vstack([row, sep, info]))
    print(f"  Saved diagnostic: {output_path}")
