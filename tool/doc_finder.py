"""Document detection — multi-strategy segmentation + line-snap refinement.

Pipeline:
  1. Detect at low res (work_h): build mask per strategy (edge / saturation /
     value), contour -> convexHull -> quad (multi-epsilon approx, fallback
     minAreaRect), score candidates (rectangularity + area + center, with
     full-frame guards).
  2. Refine at high res: snap each quad edge to the strongest local gradient
     line (sampled along the edge normal), re-intersect lines -> sub-pixel
     corners.

All quads are float32 (4, 2) in TL/TR/BR/BL order, image coordinates.
"""
import cv2
import numpy as np

METHODS = ('edge', 'saturation', 'value')


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Return corners in TL, TR, BR, BL order."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array([pts[s.argmin()], pts[d.argmin()],
                     pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = float(np.median(gray))
    return cv2.Canny(gray, int(max(0.0, (1 - sigma) * v)),
                           int(min(255.0, (1 + sigma) * v)))


def _feature_channel(img_bgr: np.ndarray, method: str) -> np.ndarray:
    """Channel where the document boundary contrasts most for this method."""
    if method == 'edge':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1] if method == 'saturation' else hsv[:, :, 2]


def build_mask(small_bgr: np.ndarray, method: str,
               invert: bool = False) -> np.ndarray:
    """Binary mask, document = 255. For S/V the polarity is explicit (invert);
    candidates from both polarities compete by score downstream."""
    if method == 'edge':
        gray  = cv2.bilateralFilter(_feature_channel(small_bgr, 'edge'), 9, 75, 75)
        edges = _auto_canny(gray)
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    ch    = _feature_channel(small_bgr, method)
    ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(ch, 0, 255, ttype + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((9, 9),  np.uint8))
    return mask


def _quad_from_contour(c: np.ndarray) -> np.ndarray:
    """Multi-epsilon approxPolyDP, prefer 4-corner convex; fallback minAreaRect."""
    peri = cv2.arcLength(c, True)
    for f in np.linspace(0.01, 0.10, 19):
        ap = cv2.approxPolyDP(c, f * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            return ap.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def score_quad(quad: np.ndarray, shape: tuple) -> float:
    """rectangularity + area + centeredness; guards against noise & full-frame."""
    H, W = shape[:2]
    area = cv2.contourArea(quad)
    ar   = area / (H * W)
    if ar < 0.04 or ar > 0.90:
        return -1.0
    (_, (rw, rh), _) = cv2.minAreaRect(quad)
    if min(rw, rh) < 1 or max(rw, rh) / min(rw, rh) > 5.0:
        return -1.0                             # thin sliver: not a document
    rect_area = cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(quad)))
    rectangularity = area / rect_area if rect_area > 0 else 0.0
    cx, cy = quad.mean(axis=0) / [W, H]
    center = 1.0 - abs(cx - 0.5) - abs(cy - 0.5)
    score  = 0.55 * rectangularity + 0.25 * ar + 0.20 * center
    score -= 2.0 * max(0.0, ar - 0.70)          # soft penalty: near-full-frame
    # border-contact: fraction of quad perimeter glued to image borders.
    # Background regions hug borders; a real document rarely does.
    m  = 0.02 * min(W, H)
    ts = np.linspace(0.0, 1.0, 12, endpoint=False)[:, None]
    per = np.concatenate([quad[i] + ts * (quad[(i + 1) % 4] - quad[i])
                          for i in range(4)])
    on_border = float(((per[:, 0] < m) | (per[:, 0] > W - m) |
                       (per[:, 1] < m) | (per[:, 1] > H - m)).mean())
    return score - 0.5 * on_border


def _best_quad_in_mask(mask: np.ndarray, shape: tuple):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_s = None, -1.0
    for c in cnts:
        if cv2.contourArea(c) < 0.04 * shape[0] * shape[1]:
            continue
        q = _quad_from_contour(cv2.convexHull(c))
        s = score_quad(q, shape)
        if s > best_s:
            best, best_s = q, s
    return best, best_s


def _best_quad_for_method(small: np.ndarray, method: str):
    """Best candidate for a method; S/V try both Otsu polarities."""
    inverts = (False,) if method == 'edge' else (False, True)
    best, best_s = None, -1.0
    for inv in inverts:
        q, s = _best_quad_in_mask(build_mask(small, method, invert=inv),
                                  small.shape)
        if s > best_s:
            best, best_s = q, s
    return best, best_s


def _intersect(line_a, line_b):
    """Intersection of two parametric lines (p, v). None if near-parallel."""
    p, v = line_a
    q, w = line_b
    cross = v[0] * w[1] - v[1] * w[0]
    if abs(cross) < 1e-6:
        return None
    t = ((q[0] - p[0]) * w[1] - (q[1] - p[1]) * w[0]) / cross
    return p + t * v


def _fit_line(pts: np.ndarray):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32),
                                 cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0]), np.array([vx, vy])


def _inward_normal(p0: np.ndarray, p1: np.ndarray, centroid: np.ndarray):
    """Unit normal of edge p0->p1 pointing toward the quad centroid."""
    edge_v = p1 - p0
    n = np.array([-edge_v[1], edge_v[0]]) / max(float(np.linalg.norm(edge_v)), 1e-6)
    mid = (p0 + p1) / 2
    return n if float(np.dot(centroid - mid, n)) > 0 else -n


def _mask_agreement(mask: np.ndarray, pts: np.ndarray,
                    n_in: np.ndarray, probe: float = 8.0) -> float:
    """Fraction of pts where the document mask is 255 just inside the line and
    0 just outside — discriminates the true boundary from parallel structures
    (e.g. outer bezel edge: dark on both sides -> agreement ~0)."""
    H, W = mask.shape
    def at(p):
        xs = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
        return mask[ys, xs] > 0
    return float((at(pts + probe * n_in) & ~at(pts - probe * n_in)).mean())


def _profile_snap_line(gx: np.ndarray, gy: np.ndarray,
                       p0: np.ndarray, p1: np.ndarray,
                       band: int, n_samples: int, min_grad: float,
                       mask: np.ndarray | None = None,
                       centroid: np.ndarray | None = None):
    """Local edge search: directional gradient profile along the edge normal.

    Only the gradient component along the normal counts (|grad . n|) — texture
    and text gradients at other orientations are suppressed. Peak offsets get
    a Gaussian proximity weight (predictable snap: prefer the *near* edge over
    a stronger far one) and parabolic sub-pixel interpolation.
    Returns (origin, direction, confidence 0..1) or None.
    """
    H, W = gx.shape
    edge_v = p1 - p0
    length = float(np.linalg.norm(edge_v))
    if length < 4:
        return None
    n = np.array([-edge_v[1], edge_v[0]]) / length          # unit normal
    offs   = np.arange(-band, band + 1)
    w_near = np.exp(-0.5 * (offs / (0.6 * band)) ** 2)
    sgn = 0.0                                    # sign of n toward quad inside
    if mask is not None and centroid is not None:
        mid = (p0 + p1) / 2
        sgn = 1.0 if float(np.dot(centroid - mid, n)) > 0 else -1.0
    probe = 8
    snapped = []
    for t in np.linspace(0.10, 0.90, n_samples):
        base = p0 + t * edge_v
        xs = np.clip(np.round(base[0] + offs * n[0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(base[1] + offs * n[1]).astype(int), 0, H - 1)
        prof = np.abs(gx[ys, xs] * n[0] + gy[ys, xs] * n[1])
        w = w_near
        if sgn != 0.0 and mask is not None:      # boundary must have document
            xi = np.clip(np.round(base[0] + (offs + sgn * probe) * n[0]).astype(int), 0, W - 1)
            yi = np.clip(np.round(base[1] + (offs + sgn * probe) * n[1]).astype(int), 0, H - 1)
            xo = np.clip(np.round(base[0] + (offs - sgn * probe) * n[0]).astype(int), 0, W - 1)
            yo = np.clip(np.round(base[1] + (offs - sgn * probe) * n[1]).astype(int), 0, H - 1)
            ok = (mask[yi, xi] > 0) & (mask[yo, xo] == 0)   # inside doc, outside not
            w = w_near * (0.25 + 0.75 * ok)
        k = int((prof * w).argmax())
        if prof[k] < min_grad:
            continue
        delta = 0.0
        if 0 < k < len(offs) - 1:                           # parabolic sub-pixel
            denom = prof[k - 1] - 2 * prof[k] + prof[k + 1]
            if abs(denom) > 1e-6:
                delta = float(np.clip(0.5 * (prof[k - 1] - prof[k + 1]) / denom,
                                      -1.0, 1.0))
        snapped.append(base + (offs[k] + delta) * n)
    if len(snapped) < max(4, n_samples // 3):
        return None
    pts = np.array(snapped, dtype=np.float32)
    o, d = _fit_line(pts)
    resid   = np.abs((pts[:, 0] - o[0]) * d[1] - (pts[:, 1] - o[1]) * d[0])
    inlier  = float((resid < 2.5).mean())
    return o, d, inlier * len(snapped) / n_samples


def _hough_snap_line(canny: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                     band: int, angle_tol_deg: float = 8.0,
                     mask: np.ndarray | None = None,
                     centroid: np.ndarray | None = None):
    """Global edge search: Hough segments inside a strip around the edge.

    Aggregates collinear evidence (works across gaps/occlusion), filters by
    angle, clusters parallel candidates by signed offset and snaps to the
    nearest sufficiently-supported cluster — not the strongest, so a parallel
    structure (bezel, taskbar) further away never steals the edge.
    Returns (origin, direction, confidence 0..1) or None.
    """
    edge_v = p1 - p0
    length = float(np.linalg.norm(edge_v))
    if length < 4:
        return None
    u = edge_v / length
    n = np.array([-u[1], u[0]])

    strip = np.zeros_like(canny)
    ext   = (0.08 * length) * u
    a = tuple(np.round(p0 - ext).astype(int))
    b = tuple(np.round(p1 + ext).astype(int))
    cv2.line(strip, a, b, 255, thickness=max(3, 2 * band))
    strip = cv2.bitwise_and(canny, strip)

    segs = cv2.HoughLinesP(strip, 1, np.pi / 360, threshold=25,
                           minLineLength=max(15.0, 0.15 * length),
                           maxLineGap=max(6.0, 0.04 * length))
    if segs is None:
        return None

    cos_tol = np.cos(np.deg2rad(angle_tol_deg))
    cands = []                                   # (offset, length, pt_a, pt_b)
    for x1, y1, x2, y2 in segs[:, 0].astype(float):
        sv = np.array([x2 - x1, y2 - y1])
        sl = float(np.linalg.norm(sv))
        if sl < 1 or abs(float(np.dot(sv / sl, u))) < cos_tol:
            continue
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        off = float(np.dot(mid - p0, n))         # signed distance to edge line
        cands.append((off, sl, (x1, y1), (x2, y2)))
    if not cands:
        return None

    cands.sort(key=lambda c: c[0])               # cluster by offset, gap > 4px
    clusters, cur = [], [cands[0]]
    for prev, c in zip(cands, cands[1:]):
        if c[0] - prev[0] > 4.0:
            clusters.append(cur)
            cur = []
        cur.append(c)
    clusters.append(cur)

    good = [cl for cl in clusters
            if sum(s[1] for s in cl) >= 0.30 * length]      # enough support
    if not good:
        return None

    n_in = None
    if mask is not None and centroid is not None:
        n_in = _inward_normal(p0, p1, centroid)

    def cl_score(cl):                            # support x proximity x mask:
        support = sum(s[1] for s in cl)          # a strong line slightly further
        off     = abs(float(np.mean([s[0] for s in cl])))   # beats a weak near one
        score   = support * np.exp(-off / band)
        if n_in is not None and mask is not None:
            pts   = np.array([p for s in cl for p in (s[2], s[3])], dtype=np.float32)
            agree = _mask_agreement(mask, pts, n_in)
            score *= 0.25 + 0.75 * agree         # parallel non-boundary lines lose
        return score
    best = max(good, key=cl_score)
    pts  = np.array([p for s in best for p in (s[2], s[3])], dtype=np.float32)
    o, d = _fit_line(pts)
    conf = min(1.0, sum(s[1] for s in best) / length)
    return o, d, conf


def _find_edge_line(channels, p0, p1, band, n_samples, min_grad,
                    mask=None, centroid=None):
    """Two-tier edge search across channels: global Hough evidence first,
    directional gradient profile as fallback. The boundary may live in a
    different channel than the segmentation mask (e.g. screen vs wall has no
    saturation gradient), so every channel competes by confidence.
    Returns (origin, direction) or None."""
    hough = [r for r in (_hough_snap_line(c, p0, p1, band,
                                          mask=mask, centroid=centroid)
                         for _gx, _gy, c in channels) if r is not None]
    best_h = max(hough, key=lambda r: r[2], default=None)
    if best_h is not None and best_h[2] >= 0.45:
        return best_h[0], best_h[1]
    prof = [r for r in (_profile_snap_line(gx, gy, p0, p1, band,
                                           n_samples, min_grad,
                                           mask=mask, centroid=centroid)
                        for gx, gy, _c in channels) if r is not None]
    best_p = max(prof, key=lambda r: r[2], default=None)
    if best_p is not None and best_p[2] >= 0.25:
        return best_p[0], best_p[1]
    if best_h is not None:                       # weak but real global evidence
        return best_h[0], best_h[1]
    return None


def _search_channels(img: np.ndarray, method: str):
    """(gx, gy, canny) for the method's feature channel plus gray fallback."""
    names = {'edge', method}                     # 'edge' == gray
    out = []
    for nm in names:
        ch = cv2.GaussianBlur(_feature_channel(img, nm), (5, 5), 0)
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        out.append((gx, gy, _auto_canny(ch)))
    return out


def _document_mask(img: np.ndarray, method: str):
    """Segmentation mask at full size (None for 'edge' — Canny lines are not a
    region). Built at detection resolution for consistency, then upscaled."""
    if method == 'edge':
        return None
    H, W = img.shape[:2]
    s = min(1.0, 600 / H)
    small = cv2.resize(img, (int(W * s), int(H * s))) if s < 1.0 else img
    m = build_mask(small, method)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)


def refine_quad(img: np.ndarray, quad: np.ndarray, method: str = 'edge',
                band: int = 14, n_samples: int = 24,
                min_grad: float = 30.0) -> np.ndarray:
    """Snap each quad edge to the document boundary within +-band px, then
    re-intersect adjacent lines for sub-pixel corners. Edges where the search
    finds nothing (or the shift is implausible) keep their input position."""
    quad = order_quad(quad)
    channels = _search_channels(img, method)
    mask     = _document_mask(img, method)
    centroid = quad.mean(axis=0)

    lines = []
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        found = _find_edge_line(channels, p0, p1, band, n_samples, min_grad,
                                mask=mask, centroid=centroid)
        if found is None:
            ev = p1 - p0
            lv = float(np.linalg.norm(ev))
            if lv < 4:
                return quad
            found = (p0, ev / lv)
        lines.append(found)

    new = []
    for i in range(4):                           # corner i = edge[i-1] x edge[i]
        c = _intersect(lines[i - 1], lines[i])
        new.append(quad[i] if c is None else c)
    new = np.array(new, dtype=np.float32)

    if np.abs(new - quad).max() > 3 * band:      # implausible jump: reject
        return quad
    return order_quad(new)


def refine_corner(img: np.ndarray, quad: np.ndarray, idx: int,
                  method: str = 'edge', band: int = 24,
                  n_samples: int = 24, min_grad: float = 30.0) -> np.ndarray:
    """Magnetic corner: re-snap only the two edges adjacent to corner idx and
    move that corner to their intersection. Other corners stay fixed."""
    quad = order_quad(quad).copy()
    channels = _search_channels(img, method)
    mask     = _document_mask(img, method)
    centroid = quad.mean(axis=0)

    lines = []
    for e in (idx - 1, idx):                     # edges meeting at corner idx
        p0, p1 = quad[e % 4], quad[(e + 1) % 4]
        found = _find_edge_line(channels, p0, p1, band, n_samples, min_grad,
                                mask=mask, centroid=centroid)
        if found is None:
            ev = p1 - p0
            lv = float(np.linalg.norm(ev))
            if lv < 4:
                return quad
            found = (p0, ev / lv)
        lines.append(found)

    c = _intersect(lines[0], lines[1])
    if c is not None and float(np.linalg.norm(c - quad[idx])) <= 2.5 * band:
        quad[idx] = c
    return quad


def find_document(img: np.ndarray, work_h: int = 600, method: str = 'auto',
                  refine: bool = True, refine_h: int = 1600):
    """Detect document quad. Returns (quad | None, info).

    info = {'method', 'score', 'candidates': {m: (quad | None, score)}}
    Quads are in full-resolution image coordinates.
    """
    H, W  = img.shape[:2]
    scale = work_h / H
    small = cv2.resize(img, (int(W * scale), work_h))

    methods = METHODS if method == 'auto' else (method,)
    candidates = {}
    for m in methods:
        q, s = _best_quad_for_method(small, m)
        candidates[m] = (None if q is None else order_quad(q) / scale, s)

    best_m = max(candidates, key=lambda m: candidates[m][1])
    quad, score = candidates[best_m]
    info = {'method': best_m, 'score': score, 'candidates': candidates}
    if quad is None:
        return None, info

    if refine:
        rs   = min(1.0, refine_h / H)
        rimg = cv2.resize(img, (int(W * rs), int(H * rs))) if rs < 1.0 else img
        quad = refine_quad(rimg, quad * rs, method=best_m) / rs

    return order_quad(quad), info


def warp_document(img: np.ndarray, quad: np.ndarray,
                  enhance: str = 'none') -> np.ndarray:
    """Perspective-warp quad to a flat image. enhance: 'none' | 'bw' | 'clahe'."""
    quad = order_quad(quad)
    tl, tr, br, bl = quad
    out_w = max(int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))), 8)
    out_h = max(int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))), 8)
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M    = cv2.getPerspectiveTransform(quad, dst)
    warp = cv2.warpPerspective(img, M, (out_w, out_h))

    if enhance == 'bw':
        g    = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        bw   = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 10)
        warp = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    elif enhance == 'clahe':
        lab     = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        warp    = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return warp
