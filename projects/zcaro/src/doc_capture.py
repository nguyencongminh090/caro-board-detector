"""Document Capture — auto-detect quad, drag corners to refine, warp preview.

Workflow: Open image -> auto-detect suggests 4 corners -> drag to adjust
(or Snap to edges for sub-pixel line fitting) -> Save warped scan to image/.
"""
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QCheckBox, QGroupBox, QGridLayout, QSizePolicy,
    QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # projects/zcaro -> src.*
sys.path.insert(0, str(_HERE.parents[3]))   # CV-Workspace   -> common.*
from src.doc_finder import (METHODS, find_document, refine_quad,
                            refine_corner, warp_document, order_quad)

HANDLE_HIT_PX = 16          # grab radius around a corner, in widget pixels
DISP_H        = 1000        # editor working resolution (display + preview warp)
REFINE_H      = 1600        # resolution for snap-to-edges refinement
SEARCH_BANDS  = {'Near': 16, 'Medium': 32, 'Far': 64}   # snap range, refine px
PAD_FRAC      = 0.12        # editor margin (so off-frame corners stay grabbable)


class _Canvas(QLabel):
    """Scalable image canvas emitting mouse events in image coordinates."""
    mouse_pressed  = pyqtSignal(float, float)
    mouse_moved    = pyqtSignal(float, float)
    mouse_released = pyqtSignal(float, float)

    def __init__(self, interactive: bool = False, placeholder: str = '') -> None:
        super().__init__(placeholder)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet('background: #111; color: #555; font-size: 13px;')
        self._orig_pix: QPixmap | None = None
        self._img_ref:  np.ndarray | None = None
        self._content_pad = 0       # pad (in shown-image px) framing the content
        if interactive:
            self.setMouseTracking(True)

    def set_content_pad(self, pad: float) -> None:
        """Pixels of border the shown image carries around real content.

        Mouse coords are reported in the *content* frame (pad subtracted), so a
        click in the border region yields negative / out-of-range coordinates —
        exactly what lets an off-frame corner be grabbed at its true position.
        """
        self._content_pad = max(0, int(pad))

    def show_bgr(self, img: np.ndarray) -> None:
        h, w          = img.shape[:2]
        self._img_ref = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg          = QImage(self._img_ref.data, w, h, 3 * w,  # type: ignore[call-overload]
                               QImage.Format.Format_RGB888)
        self._orig_pix = QPixmap.fromImage(qimg)
        self._redraw()

    def clear_canvas(self, msg: str = '') -> None:
        self._orig_pix = None
        self._img_ref  = None
        self.setText(msg)

    def view_scale(self) -> float:
        """Widget pixels per image pixel at current size (0 if no image)."""
        if self._orig_pix is None:
            return 0.0
        iw, ih = self._orig_pix.width(), self._orig_pix.height()
        if iw == 0 or ih == 0:
            return 0.0
        return min(self.width() / iw, self.height() / ih)

    def _redraw(self) -> None:
        if self._orig_pix is None:
            return
        self.setText('')
        self.setPixmap(self._orig_pix.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, a0) -> None:  # noqa: ARG002
        self._redraw()

    def _img_xy(self, pos, clamp: bool = False) -> tuple[float, float] | None:
        if self._orig_pix is None:
            return None
        s = self.view_scale()
        if s <= 0:
            return None
        iw, ih = self._orig_pix.width(), self._orig_pix.height()
        ox = (self.width()  - iw * s) / 2
        oy = (self.height() - ih * s) / 2
        x  = (pos.x() - ox) / s     # shown-image px (content + pad border)
        y  = (pos.y() - oy) / s
        p  = self._content_pad
        if clamp:                    # keep within the padded frame
            x = min(max(x, 0.0), iw - 1.0)
            y = min(max(y, 0.0), ih - 1.0)
            return x - p, y - p
        if 0 <= x < iw and 0 <= y < ih:
            return x - p, y - p      # content-frame coords (may be negative)
        return None

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            xy = self._img_xy(ev.pos())
            if xy:
                self.mouse_pressed.emit(*xy)

    def mouseMoveEvent(self, ev) -> None:
        xy = self._img_xy(ev.pos(), clamp=True)   # keep dragging at borders
        if xy:
            self.mouse_moved.emit(*xy)

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            xy = self._img_xy(ev.pos(), clamp=True)
            if xy:
                self.mouse_released.emit(*xy)


class DocCapture(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Document Capture — auto detect + drag corners')
        self._img_full:   np.ndarray | None = None
        self._img_disp:   np.ndarray | None = None
        self._img_refine: np.ndarray | None = None   # cached <=REFINE_H copy
        self._disp_scale:   float = 1.0
        self._refine_scale: float = 1.0
        self._quad:       np.ndarray | None = None   # full-res coords, TL/TR/BR/BL
        self._auto_quad:  np.ndarray | None = None   # last auto result (for Reset)
        self._feat_method = 'edge'                   # channel used by Snap
        self._drag_idx:   int | None = None
        self._build_ui()

    # ----------------------------------------------------------------- layout
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Editor canvas
        ev_ = QVBoxLayout()
        lbl_e = QLabel('Editor  (drag corners)')
        lbl_e.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_e.setStyleSheet('color: #777; font-size: 11px;')
        self._editor = _Canvas(interactive=True, placeholder='Open an image…')
        self._editor.mouse_pressed.connect(self._on_press)
        self._editor.mouse_moved.connect(self._on_move)
        self._editor.mouse_released.connect(self._on_release)
        ev_.addWidget(lbl_e)
        ev_.addWidget(self._editor)
        root.addLayout(ev_, stretch=3)

        # Warp preview canvas
        pv_ = QVBoxLayout()
        lbl_p = QLabel('Warped preview')
        lbl_p.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_p.setStyleSheet('color: #777; font-size: 11px;')
        self._preview = _Canvas(interactive=False, placeholder='—')
        pv_.addWidget(lbl_p)
        pv_.addWidget(self._preview)
        root.addLayout(pv_, stretch=2)

        # Right panel
        panel = QWidget()
        panel.setFixedWidth(250)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(4, 4, 4, 4)
        pv.setSpacing(6)
        root.addWidget(panel)

        btn_open = QPushButton('📂  Open image…')
        btn_open.setFixedHeight(36)
        btn_open.setStyleSheet('font-weight: bold; font-size: 13px;')
        btn_open.clicked.connect(self._open_image)
        pv.addWidget(btn_open)

        # Detection
        grp_d = QGroupBox('Detection')
        dg = QGridLayout(grp_d)
        self._cmb_method = QComboBox()
        self._cmb_method.addItems(['auto', *METHODS])
        self._btn_detect = QPushButton('Detect')
        self._btn_detect.clicked.connect(self._detect)
        self._lbl_method = QLabel('—')
        self._lbl_score  = QLabel('—')
        dg.addWidget(QLabel('Method:'),  0, 0)
        dg.addWidget(self._cmb_method,   0, 1)
        dg.addWidget(self._btn_detect,   1, 0, 1, 2)
        dg.addWidget(QLabel('Used:'),    2, 0)
        dg.addWidget(self._lbl_method,   2, 1)
        dg.addWidget(QLabel('Score:'),   3, 0)
        dg.addWidget(self._lbl_score,    3, 1)
        pv.addWidget(grp_d)

        # Corners
        grp_c = QGroupBox('Corners')
        cg = QVBoxLayout(grp_c)
        rng_row = QHBoxLayout()
        rng_row.addWidget(QLabel('Search range:'))
        self._cmb_range = QComboBox()
        self._cmb_range.addItems(list(SEARCH_BANDS))
        self._cmb_range.setCurrentText('Medium')
        self._cmb_range.setToolTip('How far around each side/corner the edge search looks')
        rng_row.addWidget(self._cmb_range, stretch=1)
        cg.addLayout(rng_row)
        self._chk_magnet = QCheckBox('Magnetic corners')
        self._chk_magnet.setChecked(True)
        self._chk_magnet.setToolTip(
            'After dragging a corner, snap it to the nearest edge intersection')
        cg.addWidget(self._chk_magnet)
        self._btn_snap  = QPushButton('🧲  Snap to edges')
        self._btn_snap.setToolTip('Fit each side to the document boundary nearby')
        self._btn_snap.clicked.connect(self._snap)
        self._btn_reset = QPushButton('↺  Reset to auto')
        self._btn_reset.clicked.connect(self._reset_quad)
        cg.addWidget(self._btn_snap)
        cg.addWidget(self._btn_reset)
        pv.addWidget(grp_c)

        # Output
        grp_o = QGroupBox('Output')
        og = QGridLayout(grp_o)
        self._cmb_enh = QComboBox()
        self._cmb_enh.addItems(['none', 'clahe', 'bw'])
        self._cmb_enh.currentTextChanged.connect(lambda _t: self._update_preview())
        self._btn_save = QPushButton('💾  Save scan')
        self._btn_save.clicked.connect(self._save)
        self._lbl_size  = QLabel('—')
        self._lbl_saved = QLabel('')
        self._lbl_saved.setWordWrap(True)
        self._lbl_saved.setStyleSheet('color: #00e676; font-size: 10px;')
        og.addWidget(QLabel('Enhance:'), 0, 0)
        og.addWidget(self._cmb_enh,      0, 1)
        og.addWidget(QLabel('Size:'),    1, 0)
        og.addWidget(self._lbl_size,     1, 1)
        og.addWidget(self._btn_save,     2, 0, 1, 2)
        og.addWidget(self._lbl_saved,    3, 0, 1, 2)
        pv.addWidget(grp_o)

        pv.addStretch()
        self._set_controls_enabled(False)
        self.resize(1280, 760)

    def _set_controls_enabled(self, on: bool) -> None:
        for w in (self._btn_detect, self._btn_snap, self._btn_reset,
                  self._btn_save, self._cmb_method, self._cmb_enh):
            w.setEnabled(on)

    # ----------------------------------------------------------------- file
    def _open_image(self) -> None:
        img_dir = Path(__file__).parent.parent / 'image'
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open image', str(img_dir),
            'Images (*.png *.jpg *.jpeg *.bmp *.webp)')
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self._lbl_saved.setText(f'Cannot read: {path}')
            return
        self._img_full = img
        H, W = img.shape[:2]
        self._disp_scale = min(1.0, DISP_H / H)
        self._img_disp = cv2.resize(
            img, (int(W * self._disp_scale),
                  int(H * self._disp_scale))) if self._disp_scale < 1.0 else img
        self._refine_scale = min(1.0, REFINE_H / H)
        self._img_refine = cv2.resize(
            img, (int(W * self._refine_scale),
                  int(H * self._refine_scale))) if self._refine_scale < 1.0 else img
        self._lbl_saved.setText('')
        self._set_controls_enabled(True)
        self._detect()

    # ----------------------------------------------------------------- detect
    def _detect(self) -> None:
        if self._img_full is None:
            return
        method = self._cmb_method.currentText()
        quad, info = find_document(self._img_full, method=method)
        if quad is None:                         # always give 4 corners to drag
            H, W = self._img_full.shape[:2]
            quad = order_quad(np.array(
                [[0.15 * W, 0.15 * H], [0.85 * W, 0.15 * H],
                 [0.85 * W, 0.85 * H], [0.15 * W, 0.85 * H]], dtype=np.float32))
            self._lbl_method.setText('not found')
            self._lbl_score.setText('—  (drag manually)')
        else:
            self._lbl_method.setText(info['method'])
            self._lbl_score.setText(f"{info['score']:.3f}")
        self._feat_method = info['method'] if info['score'] > 0 else 'edge'
        self._quad      = quad
        self._auto_quad = quad.copy()
        self._render()

    def _band(self) -> int:
        return SEARCH_BANDS[self._cmb_range.currentText()]

    def _snap(self) -> None:
        if self._img_refine is None or self._quad is None:
            return
        rs = self._refine_scale
        self._quad = refine_quad(self._img_refine, self._quad * rs,
                                 method=self._feat_method,
                                 band=self._band()) / rs
        self._render()

    def _reset_quad(self) -> None:
        if self._auto_quad is not None:
            self._quad = self._auto_quad.copy()
            self._render()

    # ----------------------------------------------------------------- drag
    def _on_press(self, x: float, y: float) -> None:
        if self._quad is None or self._img_disp is None:
            return
        s = self._editor.view_scale()
        if s <= 0:
            return
        hit_r = HANDLE_HIT_PX / s                     # widget px -> disp px
        qd    = self._quad * self._disp_scale         # corners in disp coords
        d     = np.linalg.norm(qd - np.array([x, y]), axis=1)
        i     = int(d.argmin())
        self._drag_idx = i if d[i] <= hit_r else None

    def _on_move(self, x: float, y: float) -> None:
        if self._drag_idx is None or self._quad is None:
            return
        self._quad[self._drag_idx] = (x / self._disp_scale,
                                      y / self._disp_scale)
        self._render()

    def _on_release(self, x: float, y: float) -> None:
        if self._drag_idx is None:
            return
        self._on_move(x, y)
        idx, self._drag_idx = self._drag_idx, None
        if (self._chk_magnet.isChecked()
                and self._img_refine is not None and self._quad is not None):
            rs = self._refine_scale
            self._quad = refine_corner(self._img_refine, self._quad * rs, idx,
                                       method=self._feat_method,
                                       band=self._band()) / rs
            self._render()

    # ----------------------------------------------------------------- render
    def _render(self) -> None:
        if self._img_disp is None or self._quad is None:
            return
        Hd, Wd = self._img_disp.shape[:2]
        pad    = int(round(PAD_FRAC * max(Hd, Wd)))
        # frame the content in a gray border so off-image corners stay visible
        vis = cv2.copyMakeBorder(self._img_disp, pad, pad, pad, pad,
                                 cv2.BORDER_CONSTANT, value=(40, 40, 40))
        Hc, Wc = vis.shape[:2]
        qd  = self._quad * self._disp_scale + pad        # -> padded-canvas coords
        pts = qd.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        for i, (px, py) in enumerate(qd):
            active  = (i == self._drag_idx)
            # is the true corner outside the document image itself?
            outside = not (pad <= px < Wc - pad and pad <= py < Hc - pad)
            if px < 0 or py < 0 or px >= Wc or py >= Hc:
                # beyond even the pad: clamp to border + arrow toward true spot
                cx, cy = int(np.clip(px, 4, Wc - 5)), int(np.clip(py, 4, Hc - 5))
                dx, dy = int(np.clip(px - cx, -22, 22)), int(np.clip(py - cy, -22, 22))
                cv2.arrowedLine(vis, (cx, cy), (cx + dx, cy + dy),
                                (0, 220, 255), 2, tipLength=0.4)
            else:
                cx, cy = int(px), int(py)
            fill = ((0, 140, 255) if active else
                    (0, 200, 255) if outside else (0, 0, 255))
            cv2.circle(vis, (cx, cy), 11, (255, 255, 255), 2)
            cv2.circle(vis, (cx, cy), 8, fill, -1)
            cv2.putText(vis, ('TL', 'TR', 'BR', 'BL')[i], (cx + 12, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        self._editor.set_content_pad(pad)
        self._editor.show_bgr(vis)
        self._update_preview()

    def _update_preview(self) -> None:
        if self._img_disp is None or self._quad is None:
            return
        warp = warp_document(self._img_disp, self._quad * self._disp_scale,
                             enhance=self._cmb_enh.currentText())
        self._preview.show_bgr(warp)
        # report FULL-RES output size (what Save will produce)
        tl, tr, br, bl = order_quad(self._quad)
        ow = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        oh = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        self._lbl_size.setText(f'{ow}×{oh}')

    # ----------------------------------------------------------------- save
    def _save(self) -> None:
        if self._img_full is None or self._quad is None:
            return
        warp = warp_document(self._img_full, self._quad,
                             enhance=self._cmb_enh.currentText())
        out_dir = Path(__file__).parent.parent / 'image'
        out_dir.mkdir(exist_ok=True)
        ts   = datetime.now().strftime('%Y%m%d-%H%M%S')
        path = out_dir / f'{ts}_scan.png'
        cv2.imwrite(str(path), warp)
        self._lbl_saved.setText(f'Saved: image/{path.name}')


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = DocCapture()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
