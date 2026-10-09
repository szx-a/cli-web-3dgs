"""端到端自检：用合成多视角图像跑通 frames → mask(可选) → colmap → export。

为什么需要它：真实数据集还没到，但 COLMAP 适配器（特征提取参数、mask 回退、
二进制解析、质量体检、去畸变数据准备）必须尽早验证，否则等数据来了再排错会非常痛苦。

用法：
    env\\venv\\Scripts\\python.exe scripts\\synthetic_e2e.py            # 默认 24 张环绕视图
    env\\venv\\Scripts\\python.exe scripts\\synthetic_e2e.py --views 36 --size 960
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_texture(size: int = 512, seed: int = 7) -> Image.Image:
    """生成高纹理平面图（棋盘 + 随机色块 + 圆点），保证 SIFT 有足够特征。"""
    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size, 3), dtype=np.uint8)

    # 棋盘底纹
    tile = size // 8
    for i in range(8):
        for j in range(8):
            c = 90 if (i + j) % 2 == 0 else 190
            arr[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile] = c

    # 随机色块
    for _ in range(60):
        y, x = rng.integers(0, size - 60, 2)
        h, w = rng.integers(20, 70, 2)
        arr[y:y + h, x:x + w] = rng.integers(0, 255, 3)

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    # 圆点与线条，增加角点
    for _ in range(80):
        x, y = rng.integers(0, size, 2)
        r = int(rng.integers(3, 9))
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255), width=2)
    for _ in range(30):
        x1, y1, x2, y2 = rng.integers(0, size, 4)
        draw.line([x1, y1, x2, y2], fill=(20, 220, 120), width=2)
    return img


def render_view(texture: Image.Image, plane_size: float, cam_pos: np.ndarray,
                look_at: np.ndarray, focal_px: float, out_size: int) -> Image.Image:
    """用**逐像素光线投射**渲染一块带纹理的平面（世界系 z=0，法线 +z）。

    为什么不用上一版的单应 + `PIL.Image.transform`：
    单应矩阵算出来是对的（`H @ src` 精确等于目标图像坐标），但 PIL 的
    `transform` 有自己的一套约定（fillcolor、逆映射方向、越界处理），
    结果是整块四边形被填成黑色而**不报任何错**。
    这种"库的隐式约定"最容易浪费排查时间，所以这里改成从第一性原理直接实现：
    对每个像素发一条射线，求与平面的交点，再采样纹理。逻辑完全可控、可读、可验证。

    相机约定（OpenCV / COLMAP 风格）：x 向右、y 向下、z 向前。
    """
    # --- 相机基（行 = 相机轴在世界系）---
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    cam_x = np.cross(world_up, forward)
    if np.linalg.norm(cam_x) < 1e-8:            # 正上方俯视，换一个参考轴
        cam_x = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    cam_x = cam_x / np.linalg.norm(cam_x)
    cam_y = np.cross(forward, cam_x)            # y 向下（OpenCV）
    cam_y = cam_y / np.linalg.norm(cam_y)

    # --- 像素栅格 -> 相机系射线方向 ---
    n = out_size
    c = (n - 1) / 2.0
    jj, ii = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")  # ii=行(y), jj=列(x)
    dx = (jj - c) / focal_px
    dy = (ii - c) / focal_px
    dirs = (dx[..., None] * cam_x + dy[..., None] * cam_y + forward)   # [n,n,3]
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # --- 与平面 z=0 求交 ---
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -cam_pos[2] / dirs[..., 2]
    hit = np.isfinite(t) & (t > 0)
    px = cam_pos[0] + t * dirs[..., 0]
    py = cam_pos[1] + t * dirs[..., 1]

    hs = plane_size / 2.0
    inside = hit & (np.abs(px) <= hs) & (np.abs(py) <= hs)

    # --- 采样纹理（世界 (px,py) -> 纹理 uv）---
    tex = np.asarray(texture.convert("RGB"), dtype=np.uint8)
    th, tw = tex.shape[:2]
    u = np.clip(((px / plane_size) + 0.5) * (tw - 1), 0, tw - 1).astype(np.int32)
    v = np.clip(((0.5 - py / plane_size)) * (th - 1), 0, th - 1).astype(np.int32)
    sampled = tex[v, u]                        # [n,n,3]

    # --- 合成：背景 + 平面 + 噪声与轻微模糊（模拟真实照片）---
    bg_color = np.array([35, 38, 45], dtype=np.float32)
    out = np.where(inside[..., None], sampled.astype(np.float32), bg_color)

    rng = np.random.default_rng(abs(hash(cam_pos.tobytes())) % 2**31)
    out = out + rng.normal(0, 2.5, out.shape)
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def _quad_mask(size: tuple[int, int], quad: np.ndarray) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in quad], fill=255)
    return m


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4 点 DLT 求单应矩阵（src -> dst）。"""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    return vt[-1].reshape(3, 3)


def build_dataset(out_dir: Path, views: int, size: int, plane: float = 3.0,
                  elev_deg: float = 30.0) -> int:
    """生成环绕视角的合成数据。

    ⚠️ 这里出过一个让整条流水线"看起来全坏了"的 bug，务必理解清楚：

    平面角点是 `(±hs, ±hs, 0)` —— 即平面位于 **z=0**，法线朝向世界 **+z**。
    而最初的相机轨迹是 `(r·cosθ, 小高度, r·sinθ)`：
      * θ=0 时相机在 `(r, 0.15, 0)`，视线方向几乎与平面**平行** → 平面被侧视成一条线；
      * 相机高度只有 0.15、半径 3.6 → 仰角仅 2.4°，基线也几乎退化。
    于是渲染出来是一张几乎全黑的图（只有一条淡竖线），
    SIFT 无特征（36 张只注册了 14 张），重建自然也无从谈起。

    正确做法：相机必须**绕过平面法线**（即在包含 +z 的半空间里环绕），
    这样才始终"正对"平面；再叠加一个俯仰角让视角有变化、扩大基线。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()
    texture = make_texture(768)
    radius = plane * 1.6
    focal = size * 0.9
    elev = math.radians(elev_deg)
    written = 0
    for i in range(views):
        theta = 2 * math.pi * i / views
        # 绕平面法线（+z）环绕，并向 +z 侧抬高 -> 始终正对平面
        cam = np.array([
            radius * math.cos(elev) * math.cos(theta),
            radius * math.cos(elev) * math.sin(theta),
            radius * math.sin(elev) + 0.10 * plane * math.sin(2 * theta),
        ])
        img = render_view(texture, plane, cam, np.zeros(3), focal, size)
        if img is None:
            continue
        img.save(out_dir / f"{written:05d}.jpg", quality=92)
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="合成多视角数据 + 端到端跑通 frames/colmap/export")
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--size", type=int, default=800)
    # 相机俯仰角：太小时水平面会被侧视成一条线（见 build_dataset 的注释）
    ap.add_argument("--elev", type=float, default=30.0, help="相机俯仰角（度），默认 30")
    ap.add_argument("--stages", default="frames,colmap,export")
    ap.add_argument("--out", default=str(ROOT / "env" / "synthetic" / "images"))
    ap.add_argument("--backend", default="classical", help="mask 阶段使用的分割后端")
    ap.add_argument("--skip-mask", action="store_true", help="跳过分割阶段（无分割后端时）")
    args = ap.parse_args()

    from gsflow.config import load as load_settings
    from gsflow.events import EventBus, console_sink
    from gsflow.pipeline.orchestrator import JobSpec, Orchestrator
    from gsflow.stages import register_all  # noqa: F401

    out_dir = Path(args.out)
    n = build_dataset(out_dir, args.views, args.size, elev_deg=args.elev)
    print(f"已生成合成视图 {n} 张 -> {out_dir}")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    settings = load_settings()
    spec = JobSpec(
        images_dir=str(out_dir),
        input_path=str(out_dir),
        title="合成数据端到端自检",
        stages=stages,
        params={"mask": {"backend": args.backend, "dilate_px": 2, "feather_px": 1}},
    )
    events = EventBus()
    events.subscribe(console_sink())
    orch = Orchestrator(spec, settings=settings, events=events)
    result = orch.run()
    print("\n=== 结果 ===")
    print("job_id:", result["job_id"])
    print("status:", result["status"])
    print("job_dir:", result["job_dir"])
    for k, v in result["stages"].items():
        print(f"  {k:<8} {v.get('status')}  {v.get('message', '')}")
    return 0 if result["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
