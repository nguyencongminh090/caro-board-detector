import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QSpinBox, QCheckBox, QGroupBox, QGridLayout,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # projects/zcaro -> src.*
sys.path.insert(0, str(_HERE.parents[3]))   # CV-Workspace   -> common.*
from common.screenshot import ScreenShot, OverlayRegionSelector
from src.board_finder import find_board, compute_grid, sort_corners


class DetectionWorker(QThread):
    result_ready = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self._running  = False
        self._lock     = threading.Lock()
        self._interval = 0.2
        self._hsv_lo   = (13, 9, 195)
        self._hsv_hi   = (33, 49, 255)
        self._region:       tuple[int, int, int, int] | None = None
        self._locked:       bool = False
        self._last_corners: np.ndarray | None = None
        self._last_dbg:     dict | None = None

    def set_params(self, interval_ms: int,
                   hsv_lo: tuple[int, int, int],
                   hsv_hi: tuple[int, int, int]) -> None:
        with self._lock:
            self._interval = interval_ms / 1000
            self._hsv_lo   = hsv_lo
            self._hsv_hi   = hsv_hi

    def set_region(self, region: tuple[int, int, int, int] | None) -> None:
        with self._lock:
            self._region = region

    def set_locked(self, locked: bool) -> None:
        with self._lock:
            self._locked = locked

    def run(self) -> None:
        self._running = True
        sct = ScreenShot()
        while self._running:
            with self._lock:
                interval     = self._interval
                hsv_lo       = self._hsv_lo
                hsv_hi       = self._hsv_hi
                region       = self._region
                locked       = self._locked
                last_corners = self._last_corners
                last_dbg     = self._last_dbg
            t0  = time.time()
            img = sct.screenshot()
            if region is not None:
                rx, ry, rw, rh = region
                img = img[ry:ry + rh, rx:rx + rw]
            if locked and last_corners is not None and last_dbg is not None:
                corners, dbg = last_corners, last_dbg
            else:
                corners, dbg = find_board(img, hsv_lo=hsv_lo, hsv_hi=hsv_hi)
                with self._lock:
                    self._last_corners = corners
                    self._last_dbg     = dbg
            elapsed = time.time() - t0
            self.result_ready.emit({'img': img, 'corners': corners,
                                    'dbg': dbg, 'elapsed': elapsed})
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def stop(self) -> None:
        self._running = False


class _Canvas(QLabel):
    """Scalable image canvas with optional mouse-to-image coordinate mapping."""
    pixel_hovered = pyqtSignal(int, int)
    pixel_clicked = pyqtSignal(int, int)

    def __init__(self, interactive: bool = False,
                 placeholder: str = '') -> None:
        super().__init__(placeholder)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet('background: #111; color: #555; font-size: 13px;')
        self._orig_pix: QPixmap | None = None
        self._img_ref:  np.ndarray | None = None
        if interactive:
            self.setMouseTracking(True)

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

    def _img_xy(self, pos) -> tuple[int, int] | None:
        if self._orig_pix is None:
            return None
        pw, ph = self.width(), self.height()
        iw, ih = self._orig_pix.width(), self._orig_pix.height()
        if iw == 0 or ih == 0:
            return None
        scale = min(pw / iw, ph / ih)
        ox    = (pw - iw * scale) / 2
        oy    = (ph - ih * scale) / 2
        x     = (pos.x() - ox) / scale
        y     = (pos.y() - oy) / scale
        if 0 <= x < iw and 0 <= y < ih:
            return int(x), int(y)
        return None

    def mouseMoveEvent(self, ev) -> None:
        xy = self._img_xy(ev.pos())
        if xy:
            self.pixel_hovered.emit(*xy)

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            xy = self._img_xy(ev.pos())
            if xy:
                self.pixel_clicked.emit(*xy)


class BoardTracker(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Board Tracker — Real-time')
        self._worker    = DetectionWorker()
        self._running   = False
        self._frame_ts: list[float] = []
        self._region:   tuple[int, int, int, int] | None = None
        self._locked    = False
        # HSV picker state
        self._last_bgr:  np.ndarray | None = None
        self._last_hsv:  np.ndarray | None = None
        self._samples:   list[tuple[int, int, int]] = []
        self._sample_xy: list[tuple[int, int]]      = []
        self._worker.result_ready.connect(self._on_result)
        self._build_ui()

    # ----------------------------------------------------------------- layout
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Live capture canvas (interactive — HSV pick)
        lv = QVBoxLayout()
        lbl_l = QLabel('Live capture  (click to sample HSV)')
        lbl_l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_l.setStyleSheet('color: #777; font-size: 11px;')
        self._live = _Canvas(interactive=True, placeholder='Waiting…')
        self._live.pixel_hovered.connect(self._on_hover)
        self._live.pixel_clicked.connect(self._on_click)
        lv.addWidget(lbl_l)
        lv.addWidget(self._live)
        root.addLayout(lv, stretch=3)

        # Board crop canvas
        bv = QVBoxLayout()
        lbl_b = QLabel('Detected board')
        lbl_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_b.setStyleSheet('color: #777; font-size: 11px;')
        self._crop = _Canvas(interactive=False, placeholder='Board not detected')
        bv.addWidget(lbl_b)
        bv.addWidget(self._crop)
        root.addLayout(bv, stretch=2)

        # Right panel
        panel = QWidget()
        panel.setFixedWidth(272)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(4, 4, 4, 4)
        pv.setSpacing(6)
        root.addWidget(panel)

        # Start / Stop
        self._btn = QPushButton('▶  Start')
        self._btn.setFixedHeight(36)
        self._btn.setStyleSheet('font-weight: bold; font-size: 14px;')
        self._btn.clicked.connect(self._toggle)
        pv.addWidget(self._btn)

        # Status
        grp_s = QGroupBox('Status')
        sg = QGridLayout(grp_s)
        sg.setColumnStretch(1, 1)
        self._lbl_fps     = QLabel('—')
        self._lbl_latency = QLabel('—')
        self._lbl_board   = QLabel('—')
        self._lbl_size    = QLabel('—')
        self._lbl_lines   = QLabel('—')
        for row, (k, w) in enumerate([
            ('FPS',     self._lbl_fps),
            ('Latency', self._lbl_latency),
            ('Board',   self._lbl_board),
            ('Size',    self._lbl_size),
            ('Lines',   self._lbl_lines),
        ]):
            sg.addWidget(QLabel(k + ':'), row, 0)
            sg.addWidget(w,               row, 1)
        pv.addWidget(grp_s)

        # Capture region
        grp_rgn = QGroupBox('Capture region')
        rgv = QVBoxLayout(grp_rgn)
        rgv.setSpacing(4)
        self._lbl_region = QLabel('Full screen')
        self._lbl_region.setStyleSheet('font-family: monospace; font-size: 11px;')
        rg_row = QHBoxLayout()
        btn_sel = QPushButton('Select…')
        btn_sel.clicked.connect(self._select_region)
        btn_rst = QPushButton('Reset')
        btn_rst.clicked.connect(self._reset_region)
        rg_row.addWidget(btn_sel)
        rg_row.addWidget(btn_rst)
        rgv.addWidget(self._lbl_region)
        rgv.addLayout(rg_row)
        pv.addWidget(grp_rgn)

        # HSV Picker
        grp_pick = QGroupBox('HSV Picker')
        hpv = QVBoxLayout(grp_pick)
        hpv.setSpacing(4)

        # Swatch + value readout
        info_row = QHBoxLayout()
        self._swatch = QLabel()
        self._swatch.setFixedSize(22, 22)
        self._swatch.setStyleSheet('background: #333; border: 1px solid #555;')
        self._lbl_hsv_info = QLabel('H:—  S:—  V:—')
        self._lbl_hsv_info.setStyleSheet('font-family: monospace; font-size: 11px;')
        info_row.addWidget(self._swatch)
        info_row.addWidget(self._lbl_hsv_info, stretch=1)
        hpv.addLayout(info_row)

        # Sample controls
        sr = QHBoxLayout()
        self._lbl_n = QLabel('0 samples')
        btn_clear   = QPushButton('Clear')
        btn_clear.setFixedWidth(46)
        btn_clear.clicked.connect(self._clear_samples)
        lbl_margin  = QLabel('±')
        self._spn_margin = QSpinBox()
        self._spn_margin.setRange(0, 60)
        self._spn_margin.setValue(20)
        self._spn_margin.setFixedWidth(46)
        self._spn_margin.valueChanged.connect(self._recompute_range)
        sr.addWidget(self._lbl_n)
        sr.addStretch()
        sr.addWidget(btn_clear)
        sr.addWidget(lbl_margin)
        sr.addWidget(self._spn_margin)
        hpv.addLayout(sr)
        pv.addWidget(grp_pick)

        # Interval
        grp_int = QGroupBox('Capture interval')
        iv = QHBoxLayout(grp_int)
        self._sld_int = QSlider(Qt.Orientation.Horizontal)
        self._sld_int.setRange(50, 2000)
        self._sld_int.setValue(200)
        self._spn_int = QSpinBox()
        self._spn_int.setRange(50, 2000)
        self._spn_int.setValue(200)
        self._spn_int.setSuffix(' ms')
        self._spn_int.setFixedWidth(70)
        self._sld_int.valueChanged.connect(self._sync_int_sld)
        self._spn_int.valueChanged.connect(self._sync_int_spn)
        iv.addWidget(self._sld_int)
        iv.addWidget(self._spn_int)
        pv.addWidget(grp_int)

        # HSV range (auto-updated by picker, also editable manually)
        grp_hsv = QGroupBox('HSV range')
        hv = QGridLayout(grp_hsv)
        labels   = ['H lo', 'S lo', 'V lo', 'H hi', 'S hi', 'V hi']
        limits   = [(0,180),(0,255),(0,255),(0,180),(0,255),(0,255)]
        defaults = [13, 9, 195, 33, 49, 255]
        self._hsv_spins: list[QSpinBox] = []
        for i, (lbl, lim, val) in enumerate(zip(labels, limits, defaults)):
            spn = QSpinBox()
            spn.setRange(*lim)
            spn.setValue(val)
            spn.valueChanged.connect(self._push_params)
            self._hsv_spins.append(spn)
            row, col = divmod(i, 3)
            hv.addWidget(QLabel(lbl + ':'), row, col * 2)
            hv.addWidget(spn,               row, col * 2 + 1)
        pv.addWidget(grp_hsv)

        # Options
        self._chk_grid = QCheckBox('Show grid intersections')
        self._chk_grid.setChecked(True)
        pv.addWidget(self._chk_grid)

        self._chk_lock = QCheckBox('Lock detection')
        self._chk_lock.setToolTip('Keep capturing frames but reuse last detected board position')
        self._chk_lock.toggled.connect(self._on_lock_toggled)
        pv.addWidget(self._chk_lock)

        pv.addStretch()
        self.resize(1300, 700)

    # ----------------------------------------------------------------- control
    def _toggle(self) -> None:
        if not self._running:
            self._running = True
            self._btn.setText('■  Stop')
            self._push_params()
            self._worker.start()
        else:
            self._running = False
            self._btn.setText('▶  Start')
            self._worker.stop()

    def _sync_int_sld(self, v: int) -> None:
        self._spn_int.blockSignals(True)
        self._spn_int.setValue(v)
        self._spn_int.blockSignals(False)
        self._push_params()

    def _sync_int_spn(self, v: int) -> None:
        self._sld_int.blockSignals(True)
        self._sld_int.setValue(v)
        self._sld_int.blockSignals(False)
        self._push_params()

    def _push_params(self) -> None:
        s = self._hsv_spins
        self._worker.set_params(
            self._spn_int.value(),
            (s[0].value(), s[1].value(), s[2].value()),
            (s[3].value(), s[4].value(), s[5].value()),
        )

    # ----------------------------------------------------------------- region select
    def _select_region(self) -> None:
        self.hide()
        QTimer.singleShot(300, self._do_select_region)

    def _do_select_region(self) -> None:
        sct = ScreenShot()
        full_img = sct.screenshot()
        self._overlay = OverlayRegionSelector(full_img, sct.monitor)
        self._overlay.region_selected.connect(self._on_region_set)
        self._overlay.show()

    def _on_region_set(self, x: int, y: int, w: int, h: int) -> None:
        self.show()
        if w > 0 and h > 0:
            self._region = (x, y, w, h)
            self._worker.set_region(self._region)
            self._lbl_region.setText(f'{w}×{h}  @({x},{y})')

    def _on_lock_toggled(self, checked: bool) -> None:
        self._locked = checked
        self._worker.set_locked(checked)
        self._chk_lock.setStyleSheet('color: #ff9800; font-weight: bold;' if checked else '')

    def _reset_region(self) -> None:
        self._region = None
        self._worker.set_region(None)
        self._lbl_region.setText('Full screen')

    # ----------------------------------------------------------------- HSV picker
    def _on_hover(self, x: int, y: int) -> None:
        if self._last_hsv is None:
            return
        hv, sv, vv = (int(c) for c in self._last_hsv[y, x])
        self._lbl_hsv_info.setText(f'H:{hv:3d}  S:{sv:3d}  V:{vv:3d}')
        rgb = cv2.cvtColor(np.array([[[hv, sv, vv]]], dtype=np.uint8),
                           cv2.COLOR_HSV2RGB)[0, 0]
        self._swatch.setStyleSheet(
            f'background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid #555;'
        )

    def _on_click(self, x: int, y: int) -> None:
        if self._last_hsv is None:
            return
        hv, sv, vv = (int(c) for c in self._last_hsv[y, x])
        hsv_px: tuple[int, int, int] = (hv, sv, vv)
        self._samples.append(hsv_px)
        self._sample_xy.append((x, y))
        n = len(self._samples)
        self._lbl_n.setText(f'{n} sample{"s" if n != 1 else ""}')
        self._recompute_range()

    def _recompute_range(self) -> None:
        if not self._samples:
            return
        m  = self._spn_margin.value()
        hs = [s[0] for s in self._samples]
        ss = [s[1] for s in self._samples]
        vs = [s[2] for s in self._samples]
        new_vals = [
            max(0,   min(hs) - m), max(0,   min(ss) - m), max(0,   min(vs) - m),
            min(180, max(hs) + m), min(255, max(ss) + m), min(255, max(vs) + m),
        ]
        for spn, v in zip(self._hsv_spins, new_vals):
            spn.blockSignals(True)
            spn.setValue(v)
            spn.blockSignals(False)
        self._push_params()

    def _clear_samples(self) -> None:
        self._samples.clear()
        self._sample_xy.clear()
        self._lbl_n.setText('0 samples')

    # ----------------------------------------------------------------- detection
    def _on_result(self, data: dict) -> None:
        img_bgr: np.ndarray = data['img']
        corners             = data['corners']
        dbg: dict           = data['dbg']
        elapsed: float      = data['elapsed']

        # Store originals for HSV pick (no overlay)
        self._last_bgr = img_bgr
        self._last_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # FPS
        now = time.time()
        self._frame_ts.append(now)
        self._frame_ts = [t for t in self._frame_ts if now - t < 3.0]
        fps = (len(self._frame_ts) - 1) / (now - self._frame_ts[0] + 1e-9) \
              if len(self._frame_ts) > 1 else 0.0
        self._lbl_fps.setText(f'{fps:.1f}')
        self._lbl_latency.setText(f'{elapsed * 1000:.0f} ms')

        overlay = img_bgr.copy()

        # Draw sample dots on live view
        for sx, sy in self._sample_xy:
            cv2.circle(overlay, (sx, sy), 5, (0, 255, 0), -1)
            cv2.circle(overlay, (sx, sy), 5, (0,   0, 0),  1)

        if corners is not None:
            n  = dbg['n_lines']
            tl, tr, br, bl_ = sort_corners(corners)
            wb  = int(br[0] - tl[0])
            hb  = int(br[1] - tl[1])
            asp = min(wb, hb) / max(wb, hb) if max(wb, hb) > 0 else 0

            # Compute grid once — reused for both views
            show_grid = self._chk_grid.isChecked() and n >= 2
            grid = compute_grid(corners, n, n) if show_grid else None

            # Live view: boundary + corners + grid
            pts = corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
            for pt in corners:
                cv2.circle(overlay, tuple(pt.astype(int)), 6, (0, 0, 255), -1)
            if grid is not None:
                for r in range(n):
                    for c in range(n):
                        cv2.circle(overlay, tuple(grid[r, c].astype(int)),
                                   2, (255, 100, 0), -1)

            # Board crop view — raw crop, no overlay
            lft, top = int(tl[0]), int(tl[1])
            rgt, bot = int(br[0]), int(br[1])
            if bot > top and rgt > lft:
                self._crop.show_bgr(img_bgr[top:bot, lft:rgt])

            self._lbl_board.setStyleSheet('color: #00e676;')
            self._lbl_board.setText('Found')
            self._lbl_size.setText(f'{wb}×{hb}  (asp {asp:.3f})')
            self._lbl_lines.setText(f'{n}×{n}')
        else:
            self._crop.clear_canvas('Board not detected')
            self._lbl_board.setStyleSheet('color: #ff5252;')
            self._lbl_board.setText(f'Not found  ({dbg.get("stage", "?")})')
            self._lbl_size.setText('—')
            self._lbl_lines.setText('—')

        self._live.show_bgr(overlay)

    def closeEvent(self, e) -> None:
        self._worker.stop()
        self._worker.wait(2000)
        e.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = BoardTracker()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
