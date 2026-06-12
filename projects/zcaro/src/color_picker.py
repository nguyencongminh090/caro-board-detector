"""
tool/color_picker.py — Lấy mẫu màu HSV để dùng cho color filter trong find_board().

Cách dùng:
    python tool/color_picker.py [đường_dẫn_ảnh]
    python tool/color_picker.py          # tự mở screenshot.png nếu có

Workflow:
    1. Mở ảnh → click vùng bàn cờ để lấy mẫu màu
    2. Xem mask preview (Overlay / Mask)
    3. Điều chỉnh margin hoặc kéo slider
    4. Bấm "Copy code" → paste vào find_board(hsv_lo=..., hsv_hi=...)
"""

import os
import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QSlider, QSpinBox,
    QFileDialog, QStatusBar, QGridLayout, QSizePolicy, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage


class ImageCanvas(QLabel):
    """Hiển thị ảnh, emit toạ độ pixel khi hover / click."""

    pixel_hovered = pyqtSignal(int, int)
    pixel_clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumSize(500, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setStyleSheet('background: #1e1e1e;')
        self._orig_pix: QPixmap | None = None
        self._img_w = self._img_h = 0
        self._sx = self._sy = 1.0

    def load_bgr(self, img: np.ndarray):
        self._img_h, self._img_w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q = QImage(rgb.data, self._img_w, self._img_h,
                   rgb.strides[0], QImage.Format.Format_RGB888)
        self._orig_pix = QPixmap.fromImage(q)
        self._redraw()

    def _redraw(self):
        if self._orig_pix is None:
            return
        scaled = self._orig_pix.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)
        if self._img_w and self._img_h:
            self._sx = scaled.width()  / self._img_w
            self._sy = scaled.height() / self._img_h

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._redraw()

    def _img_xy(self, pos) -> tuple[int, int] | None:
        if not self._orig_pix or not self._img_w:
            return None
        x, y = int(pos.x() / self._sx), int(pos.y() / self._sy)
        if 0 <= x < self._img_w and 0 <= y < self._img_h:
            return x, y
        return None

    def mouseMoveEvent(self, e):
        c = self._img_xy(e.pos())
        if c:
            self.pixel_hovered.emit(*c)

    def mousePressEvent(self, e):
        c = self._img_xy(e.pos())
        if not c:
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self.pixel_clicked.emit(*c)


class ColorPicker(QMainWindow):
    """Cửa sổ chính — sample màu, tính range, preview mask."""

    MARGIN_DEFAULT = 20

    def __init__(self, image_path: str | None = None):
        super().__init__()
        self.img_bgr:  np.ndarray | None = None
        self.img_hsv:  np.ndarray | None = None
        self.samples:  list[tuple[int, int, int]] = []   # (H, S, V)
        self.sample_xy: list[tuple[int, int]]    = []    # toạ độ gốc để vẽ dot
        self.margin    = self.MARGIN_DEFAULT
        # H_lo H_hi  S_lo S_hi  V_lo V_hi
        self._range    = [0, 180, 0, 255, 0, 255]

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self._do_refresh)

        self._init_ui()
        self.setWindowTitle('HSV Color Range Picker — Zcaro')
        self.resize(1200, 720)

        if image_path and os.path.isfile(image_path):
            self._load(image_path)

    # ------------------------------------------------------------------ UI

    def _init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        hl = QHBoxLayout(root)
        hl.setContentsMargins(4, 4, 4, 4)
        hl.setSpacing(6)

        # ---------- canvas (trái) ----------
        self.canvas = ImageCanvas()
        self.canvas.pixel_hovered.connect(self._on_hover)
        self.canvas.pixel_clicked.connect(self._on_click)
        hl.addWidget(self.canvas, stretch=3)

        # ---------- panel (phải) ----------
        panel = QWidget()
        panel.setFixedWidth(310)
        pv = QVBoxLayout(panel)
        pv.setSpacing(8)
        pv.setContentsMargins(2, 2, 2, 2)
        hl.addWidget(panel)

        # Mở ảnh + chế độ xem
        row0 = QHBoxLayout()
        btn_open = QPushButton('📂 Mở ảnh')
        btn_open.clicked.connect(self._open_file)
        row0.addWidget(btn_open)
        self.combo_view = QComboBox()
        self.combo_view.addItems(['Gốc', 'Overlay mask', 'Chỉ mask'])
        self.combo_view.currentIndexChanged.connect(self._schedule_refresh)
        row0.addWidget(self.combo_view)
        pv.addLayout(row0)

        # Pixel info
        grp_px = QGroupBox('Pixel tại con trỏ')
        px_l = QHBoxLayout(grp_px)
        self.lbl_swatch = QLabel()
        self.lbl_swatch.setFixedSize(44, 44)
        self.lbl_swatch.setStyleSheet('background:#888;border:1px solid #555;')
        px_l.addWidget(self.lbl_swatch)
        self.lbl_info = QLabel('(hover lên ảnh)')
        self.lbl_info.setWordWrap(True)
        px_l.addWidget(self.lbl_info)
        pv.addWidget(grp_px)

        # Mẫu
        grp_samp = QGroupBox('Mẫu màu  (click ảnh để thêm)')
        sl_v = QVBoxLayout(grp_samp)
        self.lbl_samp = QLabel('0 mẫu')
        sl_v.addWidget(self.lbl_samp)
        row_samp = QHBoxLayout()
        btn_clear = QPushButton('Xóa mẫu')
        btn_clear.clicked.connect(self._clear_samples)
        row_samp.addWidget(btn_clear)
        self.lbl_margin = QLabel(f'margin ±{self.margin}')
        row_samp.addWidget(self.lbl_margin)
        sl_v.addLayout(row_samp)
        row_m = QHBoxLayout()
        for d in (-5, +5, +10):
            b = QPushButton(f'{d:+d}')
            b.setFixedWidth(50)
            b.clicked.connect(lambda _, dd=d: self._adj_margin(dd))
            row_m.addWidget(b)
        row_m.addStretch()
        sl_v.addLayout(row_m)
        pv.addWidget(grp_samp)

        # HSV range
        grp_r = QGroupBox('HSV Range')
        gr_l = QGridLayout(grp_r)
        gr_l.setSpacing(3)
        labels  = ['H lo', 'H hi', 'S lo', 'S hi', 'V lo', 'V hi']
        maxvals = [180,     180,    255,     255,    255,    255  ]
        initv   = self._range
        self._sliders:   list[QSlider]  = []
        self._spinboxes: list[QSpinBox] = []
        for i, (lbl, mx, iv) in enumerate(zip(labels, maxvals, initv)):
            gr_l.addWidget(QLabel(lbl), i, 0)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, mx)
            sl.setValue(iv)
            self._sliders.append(sl)
            gr_l.addWidget(sl, i, 1)
            sb = QSpinBox()
            sb.setRange(0, mx)
            sb.setValue(iv)
            sb.setFixedWidth(58)
            self._spinboxes.append(sb)
            gr_l.addWidget(sb, i, 2)
            sl.valueChanged.connect(lambda v, j=i: self._range_change(j, v, False))
            sb.valueChanged.connect(lambda v, j=i: self._range_change(j, v, True))
        pv.addWidget(grp_r)

        # Copy + output
        btn_copy = QPushButton('📋  Copy code  →  find_board()')
        btn_copy.setFixedHeight(36)
        btn_copy.clicked.connect(self._copy)
        pv.addWidget(btn_copy)

        self.lbl_out = QLabel('–')
        self.lbl_out.setWordWrap(True)
        self.lbl_out.setStyleSheet(
            'font-family: monospace; color: #33cc33;'
            'background:#111; padding:4px; border-radius:3px;'
        )
        pv.addWidget(self.lbl_out)
        pv.addStretch()

        self.setStatusBar(QStatusBar())
        self._update_out_label()

    # ---------------------------------------------------------------- handlers

    def _open_file(self):
        p, _ = QFileDialog.getOpenFileName(
            self, 'Mở ảnh', '', 'Images (*.png *.jpg *.jpeg *.bmp)'
        )
        if p:
            self._load(p)

    def _load(self, path: str):
        img = cv2.imread(path)
        if img is None:
            self.statusBar().showMessage(f'Không đọc được: {path}')
            return
        self.img_bgr  = img
        self.img_hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.samples.clear()
        self.sample_xy.clear()
        self._update_samp_label()
        self._schedule_refresh()
        self.statusBar().showMessage(
            f'{os.path.basename(path)}   {img.shape[1]}×{img.shape[0]} px'
        )

    def _on_hover(self, x: int, y: int):
        if self.img_bgr is None:
            return
        b, g, r = (int(v) for v in self.img_bgr[y, x])
        h, s, v = (int(v) for v in self.img_hsv[y, x])
        self.lbl_swatch.setStyleSheet(
            f'background:rgb({r},{g},{b});border:1px solid #555;'
        )
        self.lbl_info.setText(f'BGR  ({b}, {g}, {r})\nHSV  ({h}, {s}, {v})')
        self.statusBar().showMessage(
            f'({x}, {y})   BGR=({b},{g},{r})   HSV=({h},{s},{v})'
        )

    def _on_click(self, x: int, y: int):
        if self.img_hsv is None:
            return
        hsv = tuple(int(v) for v in self.img_hsv[y, x])
        self.samples.append(hsv)          # type: ignore[arg-type]
        self.sample_xy.append((x, y))
        self._update_samp_label()
        self._recompute_range()
        self._schedule_refresh()

    def _clear_samples(self):
        self.samples.clear()
        self.sample_xy.clear()
        self._update_samp_label()
        self._schedule_refresh()

    def _adj_margin(self, delta: int):
        self.margin = max(0, min(60, self.margin + delta))
        self.lbl_margin.setText(f'margin ±{self.margin}')
        if self.samples:
            self._recompute_range()

    def _range_change(self, idx: int, val: int, from_spin: bool):
        self._range[idx] = val
        other = self._sliders[idx] if from_spin else self._spinboxes[idx]
        other.blockSignals(True)
        other.setValue(val)
        other.blockSignals(False)
        self._update_out_label()
        self._schedule_refresh()

    def _recompute_range(self):
        if not self.samples:
            return
        arr = np.array(self.samples, dtype=np.int32)
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        maxv = [180, 255, 255]
        new = []
        for i in range(3):
            new.append(int(max(0,       lo[i] - self.margin)))
            new.append(int(min(maxv[i], hi[i] + self.margin)))
        # new = [H_lo, H_hi, S_lo, S_hi, V_lo, V_hi]
        self._range = new
        for i, v in enumerate(new):
            self._sliders[i].blockSignals(True)
            self._spinboxes[i].blockSignals(True)
            self._sliders[i].setValue(v)
            self._spinboxes[i].setValue(v)
            self._sliders[i].blockSignals(False)
            self._spinboxes[i].blockSignals(False)
        self._update_out_label()

    # ---------------------------------------------------------------- display

    def _schedule_refresh(self):
        self._debounce.start()

    def _do_refresh(self):
        if self.img_bgr is None:
            return
        mode = self.combo_view.currentIndex()

        if mode == 0:
            display = self.img_bgr.copy()
        else:
            h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = self._range
            mask = cv2.inRange(
                self.img_hsv,
                np.array([h_lo, s_lo, v_lo]),
                np.array([h_hi, s_hi, v_hi]),
            )
            if mode == 2:
                display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                # Overlay: giữ nguyên vùng match, làm tối vùng không match
                display = self.img_bgr.copy()
                not_match = mask == 0
                display[not_match] = (display[not_match] * 0.3).astype(np.uint8)
                # thêm tint xanh nhẹ vào vùng match
                g_ch = display[:, :, 1].astype(np.int32)
                g_ch[mask == 255] = np.clip(g_ch[mask == 255] + 50, 0, 255)
                display[:, :, 1] = g_ch.astype(np.uint8)

        # Vẽ dot tại các điểm đã sample
        for (sx, sy) in self.sample_xy:
            cv2.circle(display, (sx, sy), 6, (0, 255, 0),  -1)
            cv2.circle(display, (sx, sy), 7, (255, 255, 255), 1)

        self.canvas.load_bgr(display)

    # ---------------------------------------------------------------- output

    def _update_samp_label(self):
        n = len(self.samples)
        self.lbl_samp.setText(f'{n} mẫu  —  click ảnh để thêm, Xóa để reset')

    def _update_out_label(self):
        h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = self._range
        self.lbl_out.setText(
            f'hsv_lo=({h_lo}, {s_lo}, {v_lo})\n'
            f'hsv_hi=({h_hi}, {s_hi}, {v_hi})'
        )

    def _copy(self):
        h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = self._range
        code = (
            f'hsv_lo = ({h_lo}, {s_lo}, {v_lo})\n'
            f'hsv_hi = ({h_hi}, {s_hi}, {v_hi})'
        )
        QApplication.clipboard().setText(code)
        self.statusBar().showMessage('✓ Đã copy vào clipboard!')


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Mở screenshot.png mặc định nếu không có arg
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_img  = os.path.join(project_root, 'screenshot.png')

    path = sys.argv[1] if len(sys.argv) > 1 else (
        default_img if os.path.isfile(default_img) else None
    )

    w = ColorPicker(path)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
