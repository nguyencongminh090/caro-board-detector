from datetime import datetime
import os
from typing import Tuple, Optional
import sys

import cv2
import mss
from mss import MSS
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMainWindow
)
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QImage, QPixmap


class ScreenShot:
    """Utility class for high-performance screen capture using mss.

    Provides methods to capture the entire screen or specific regions,
    optimized for use in computer vision pipelines.
    """

    def __init__(self):
        """Initializes the mss instance and identifies the primary monitor."""
        self.sct     = MSS()
        self.monitor = self.sct.monitors[0]

    def get_screen_size(self) -> Tuple[int, int]:
        """Gets the total width and height of the combined screen area.

        Returns:
            Tuple[int, int]: A tuple containing (width, height).
        """
        return self.monitor["width"], self.monitor["height"]

    def screenshot(self) -> np.ndarray:
        """Captures the entire screen area.

        Returns:
            np.ndarray: BGR image array (OpenCV compatible).
        """
        sct_img = self.sct.grab(self.monitor)
        img     = np.array(sct_img)
        
        # Under Wayland, mss captures a black screen (sum of pixels is 0)
        if img.sum() == 0:
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                tmp_path = tmpfile.name
            
            try:
                # Use KDE's spectacle to capture the screen in background mode
                subprocess.run(
                    ["spectacle", "-b", "-n", "-o", tmp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                spec_img = cv2.imread(tmp_path)
                if spec_img is not None:
                    return spec_img
            except Exception as e:
                print(f"Wayland spectacle capture failed: {e}", file=sys.stderr)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def crop(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Captures a specific region of the screen.

        Args:
            left: X-coordinate of the top-left corner.
            top: Y-coordinate of the top-left corner.
            width: Width of the region to capture.
            height: Height of the region to capture.

        Returns:
            np.ndarray: BGR image array of the specified region.
        """
        img = self.screenshot()
        return self.crop_from_array(img, left, top, width, height)

    def save(self, img: np.ndarray, filepath: str = "screenshot.png"):
        """Saves an image array to the specified file path.

        Args:
            img: The image array to save.
            filepath: The destination file path.
        """
        cv2.imwrite(filepath, img)

    def crop_from_array(self, img: np.ndarray, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Crops a region from an existing full-screen image array.

        Args:
            img: The original full-screen image array.
            left: Absolute X-coordinate of the region.
            top: Absolute Y-coordinate of the region.
            width: Width of the region.
            height: Height of the region.

        Returns:
            np.ndarray: Cropped image array.
        """
        rel_left = left - self.monitor["left"]
        rel_top  = top - self.monitor["top"]
        return img[rel_top : rel_top + height, rel_left : rel_left + width]

    def save_auto(self, img: np.ndarray) -> str:
        """Saves the image with an auto-generated timestamped filename in the 'image' directory.
        
        Returns:
            str: The full path to the saved image.
        """
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        save_dir     = os.path.join(project_root, "image")
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename  = os.path.join(save_dir, f"{timestamp}.png")
        
        self.save(img, filename)
        return filename


class OverlayRegionSelector(QWidget):
    """Full-screen overlay for interactive region selection.

    Allows the user to click and drag to define a rectangular area on the screen.
    Signals the selected coordinates upon mouse release.
    """

    # Signals absolute coordinates: (left, top, width, height)
    region_selected = pyqtSignal(int, int, int, int)

    def __init__(self, background_img_np: np.ndarray, monitor_rect: dict):
        """Initializes the overlay with a background image.

        Args:
            background_img_np: A numpy array of the screen to display as background.
            monitor_rect: Dictionary containing screen geometry (left, top, width, height).
        """
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint
        )
        self.setCursor(Qt.CursorShape.CrossCursor)

        self.monitor_rect = monitor_rect
        self.setGeometry(
            monitor_rect['left'], monitor_rect['top'],
            monitor_rect['width'], monitor_rect['height']
        )

        height, width, _ = background_img_np.shape
        bytes_per_line   = 3 * width

        # Convert background to QPixmap; keep reference to img_rgb to prevent garbage collection
        self.img_rgb = cv2.cvtColor(background_img_np, cv2.COLOR_BGR2RGB)
        q_img        = QImage(self.img_rgb.data,  # type: ignore
                        width, 
                        height, bytes_per_line, 
                        QImage.Format.Format_RGB888)
        self.background_pixmap = QPixmap.fromImage(q_img)

        self.start_point = None
        self.end_point   = None

    def paintEvent(self, a0):
        """Renders the background, dimming effect, and selection rectangle."""
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.background_pixmap)

        # Dim the background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))

        # Highlight selection
        if self.start_point and self.end_point:
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawPixmap(rect, self.background_pixmap, rect)

            # Draw selection border
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.drawRect(rect)

    def mousePressEvent(self, a0):
        """Sets the starting point for the selection rectangle."""
        if a0.button() == Qt.MouseButton.LeftButton:
            self.start_point = a0.pos()
            self.end_point   = self.start_point
            self.update()

    def mouseMoveEvent(self, a0):
        """Updates the end point as the user drags the mouse."""
        if self.start_point:
            self.end_point = a0.pos()
            self.update()

    def mouseReleaseEvent(self, a0):
        """Finalizes the selection and emits the absolute coordinates."""
        if a0.button() == Qt.MouseButton.LeftButton and self.start_point:
            self.end_point = a0.pos()
            rect = QRect(self.start_point, self.end_point).normalized()

            # Hide the overlay immediately so it's not captured in the final screenshot
            self.hide()
            # Force the UI to update and actually hide the window before proceeding
            QApplication.processEvents()

            # Convert to absolute screen coordinates
            screen_left = self.monitor_rect['left'] + rect.left()
            screen_top  = self.monitor_rect['top'] + rect.top()

            self.region_selected.emit(screen_left, screen_top, rect.width(), rect.height())
            self.close()

    def keyPressEvent(self, a0):
        """Cancels the selection if the Escape key is pressed."""
        if a0.key() == Qt.Key.Key_Escape:
            self.region_selected.emit(0, 0, 0, 0)
            self.close()


class Application(QMainWindow):
    """Main application window for the Screen Capture Tool."""

    def __init__(self):
        """Initializes the main window and UI components."""
        super().__init__()
        self.screenshot_util = ScreenShot()
        self.overlay: Optional[OverlayRegionSelector] = None
        self.last_screenshot: Optional[np.ndarray] = None
        self.init_ui()

    def init_ui(self):
        """Configures the user interface layout and widgets."""
        self.setWindowTitle('Screen Capture for CV')
        self.resize(350, 150)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_info = QLabel(
            'Click "Capture" to select a region on screen.\nPress ESC to cancel.', self
        )
        self.label_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label_info)

        self.btn_capture = QPushButton('Capture', self)
        self.btn_capture.setFixedSize(120, 40)
        self.btn_capture.clicked.connect(self.start_capture)
        layout.addWidget(self.btn_capture)

        central_widget.setLayout(layout)

    def start_capture(self):
        """Prepares the system for a screenshot by hiding the main window."""
        self.hide()
        # Delay to allow window to disappear from the display buffer
        QTimer.singleShot(300, self._perform_capture)

    def _perform_capture(self):
        """Captures the screen and initializes the selection overlay."""
        # Grab the clean screen before the overlay is shown
        self.last_screenshot = self.screenshot_util.screenshot()
        
        # Show the selector overlay using the clean screenshot as background
        self.overlay = OverlayRegionSelector(self.last_screenshot, self.screenshot_util.monitor)
        self.overlay.region_selected.connect(self.on_region_selected)
        self.overlay.show()

    def on_region_selected(self, left: int, top: int, width: int, height: int):
        """Callback for when a region is selected or selection is cancelled.

        Args:
            left: Absolute X-coordinate.
            top: Absolute Y-coordinate.
            width: Region width.
            height: Region height.
        """
        self.show()

        if width > 0 and height > 0 and self.last_screenshot is not None:
            # Delegate image processing to the utility class
            cropped_img = self.screenshot_util.crop_from_array(
                self.last_screenshot, left, top, width, height
            )
            
            self.label_info.setText(
                f"Captured: {width}x{height}\nAt: (X: {left}, Y: {top})"
            )

            # Delegate saving logic to the utility class
            filename = self.screenshot_util.save_auto(cropped_img)
            
            print(f"Captured region saved to {filename}. Shape: {cropped_img.shape}")
        else:
            self.label_info.setText("Capture canceled or invalid region.")


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = Application()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

