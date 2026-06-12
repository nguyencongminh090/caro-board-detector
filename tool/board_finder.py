import cv2
import numpy as np


def _extract_bands(idx: np.ndarray) -> list[int]:
    if idx.size == 0:
        return []
    return [int(np.mean(b)) for b in np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)]


def _trim_lattice(bands: list[int], tol: float = 0.25) -> list[int]:
    while len(bands) > 3:
        d = np.diff(bands)
        pitch = float(np.median(d))
        if   abs(d[0]  - pitch) > tol * pitch: bands = bands[1:]
        elif abs(d[-1] - pitch) > tol * pitch: bands = bands[:-1]
        else: break
    return bands


def _strip_density(dark: np.ndarray, bands: list[int], axis: str,
                   i0: int, i1: int, pad: int = 3) -> float:
    a, b = bands[i0] + pad, bands[i1] - pad
    s = dark[:, a:b] if axis == 'col' else dark[a:b, :]
    return float(s.mean() / 255) if s.size else 0.0


def sort_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def compute_grid(corners: np.ndarray, rows: int, cols: int) -> np.ndarray:
    tl, tr, br, bl = sort_corners(corners)
    grid = np.zeros((rows, cols, 2), dtype=np.float32)
    for r, h in enumerate(np.linspace(0, 1, rows)):
        left  = tl + h * (bl - tl)
        right = tr + h * (br - tr)
        for c, v in enumerate(np.linspace(0, 1, cols)):
            grid[r, c] = left + v * (right - left)
    return grid


def find_board(
    img          : np.ndarray,
    hsv_lo       : tuple[int, int, int] = (13, 9, 195),
    hsv_hi       : tuple[int, int, int] = (33, 49, 255),
    close_k      : int   = 15,
    bg_delta     : int   = 10,
    min_len_frac : float = 0.25,
) -> tuple[np.ndarray | None, dict]:
    """
    Hybrid pipeline: color ROI (stage 1) → adaptive threshold + 1D erosion (stage 2).

    Returns (corners float32[4,2] in TL/TR/BR/BL order, info dict).
    corners is None on failure; info['stage'] explains why.
    """
    # Stage 1 — color segmentation
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
    if mask.mean() / 255 < 0.05:
        return None, {'stage': 'color_fail', 'n_lines': 0}

    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {'stage': 'contour_fail', 'n_lines': 0}
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Stage 2 — long-line projection inside ROI
    gray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    bg   = int(np.median(gray))
    dark = cv2.inRange(gray, 0, bg - bg_delta)
    L    = max(int(w * min_len_frac), 10)

    vert  = cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    horiz = cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))

    rb = _extract_bands(np.where(horiz.sum(axis=1) > 0)[0])
    cb = _extract_bands(np.where(vert.sum(axis=0)  > 0)[0])
    if len(rb) < 2 or len(cb) < 2:
        return None, {'stage': 'projection_fail', 'n_lines': 0}

    rb, cb = _trim_lattice(rb), _trim_lattice(cb)

    while len(cb) > len(rb) and len(cb) > 2:
        cb = cb[1:] if _strip_density(dark, cb, 'col', 0, 1) \
             > _strip_density(dark, cb, 'col', -2, -1) else cb[:-1]
    while len(rb) > len(cb) and len(rb) > 2:
        rb = rb[1:] if _strip_density(dark, rb, 'row', 0, 1) \
             > _strip_density(dark, rb, 'row', -2, -1) else rb[:-1]

    if len(rb) < 2 or len(cb) < 2:
        return None, {'stage': 'trim_fail', 'n_lines': 0}

    pitch = float(np.median(np.r_[np.diff(rb), np.diff(cb)]))
    if pitch <= 0:
        return None, {'stage': 'pitch_fail', 'n_lines': 0}

    n_lines = round(((rb[-1] - rb[0]) + (cb[-1] - cb[0])) / (2 * pitch)) + 1
    top, bottom = y + rb[0], y + rb[-1]
    left, right = x + cb[0], x + cb[-1]
    corners = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float32,
    )
    return corners, {'stage': 'ok', 'n_lines': n_lines, 'pitch': pitch, 'roi': (x, y, w, h)}
