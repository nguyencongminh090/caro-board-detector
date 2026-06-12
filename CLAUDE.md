# Zcaro — OpenCV Project

## Project
Dự án Computer Vision dùng OpenCV + Python, tập trung vào:
- **Board detection**: Tìm bảng/vùng quan trọng từ screenshot qua contour/edge detection
- **Screen capture pipeline**: PyQt6 + mss (Wayland-compatible, fallback spectacle)
- **CV experiments**: Jupyter notebooks (test.ipynb, test2.ipynb, test3.ipynb)

## Stack
- Python, OpenCV (`cv2`), NumPy, Matplotlib
- PyQt6 (UI overlay), mss (screen capture)
- Jupyter Notebook (experiments)
- Pyright (type checking)

## Conventions
- Hàm CV thuần (no side effects) đặt trong `tool/` hoặc module riêng
- Notebook dùng để prototyping; code ổn định chuyển sang `.py`
- Ảnh input/output lưu tại `image/` với timestamp auto-generated
- `tool/screenshot.py`: `ScreenShot` class là util capture chính — không duplicate

## CV Approach (đã dùng)
- Edge detection: Canny, custom Sobel + NMS + hysteresis
- Contour: `findContours` + `approxPolyDP` + `isContourConvex`
- Board finding: filter by 4-corner convex quadrilateral, max area

## Tone cho AI
- Trả lời bằng **tiếng Việt**
- Ngắn gọn, trực tiếp, không ví dụ mơ hồ
- Khi giải thích CV: nêu lý do toán học/nguyên lý, không chỉ nêu API
- Ưu tiên fix code thực tế trong project, không tạo boilerplate mới
