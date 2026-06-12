# CV-Workspace

Monorepo cho các OpenCV / Computer Vision projects cá nhân.

## Cấu trúc

```
CV-Workspace/
├── common/          # shared utils: screenshot, viz, io, geometry
├── projects/
│   ├── zcaro/       # board tracker + document capture (PyQt6, OpenCV)
│   └── aesthetic-3d/ # 3D aesthetic field research (PyTorch + 3DGS)
└── pyrightconfig.json
```

## Convention

- Ngôn ngữ: Python 3.13
- Mỗi project có `src/` (code ổn định), `notebooks/`, `data/` (gitignored), `docs/`
- Shared code đặt trong `common/`, import bằng `from common.xxx import ...`
- `sys.path` được set trong mỗi entry-point script — không dùng relative import ngầm

## Import pattern (trong src/)

```python
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # projects/<name>  -> src.*
sys.path.insert(0, str(_HERE.parents[3]))   # CV-Workspace     -> common.*
```

## Projects

### zcaro
Board detection (real-time HSV + contour) và Document Capture (perspective warp, snap-to-edge).
Entry points: `src/board_tracker.py`, `src/doc_capture.py`

### aesthetic-3d
Research: 3D Aesthetic Field (paper 008). Stack: PyTorch, DepthSplat, gsplat.
