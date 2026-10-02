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
    """用「平面单应」把纹理投影到相机平面：精确、可控，适合做 SfM 的合成输入。

    相机系约定：右手，x 向右，y 向上，z 朝后（OpenCV 风格）。
    相机的“上”统一由世界 up 轴推出（而不是每帧各自 cross），
    这样各视角共用一个水平面，SfM 才有稳定的解。
    """
    world_up = np.array([0.0, 1.0, 0.0])
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)

    # 相机基（世界 -> 相机）
    R = np.stack([right, up, forward], axis=0)  # 行 = 相机轴在世界系
    t = -R @ cam_pos

    w, h = out_size, out_size
    cx, cy = w / 2, h / 2
    K = np.array([[focal_px, 0, cx], [0, focal_px, cy], [0, 0, 1.0]])

    # 平面四角（世界系，z=0，中心在原点）
    hs = plane_size / 2
    corners_world = np.array([
        [-hs, -hs, 0.0],
        [hs, -hs, 0.0],
        [hs, hs, 0.0],
        [-hs, hs, 0.0],
    ])
    corners_cam = (R @ corners_world.T).T + t
    if np.any(corners_cam[:, 2] <= 0.05):
        return None  # 有角点在相机后方，跳过

    corners_img = (K @ corners_cam.T).T
    corners_img = corners_img[:, :2] / corners_img[:, 2:3]

    # 目标图上的四个角（顺序与上面一致）
    dst = corners_img.astype(np.float32)
    src = np.array([[0, 0], [texture.width, 0], [texture.width, texture.height], [0, texture.height]], dtype=np.float32)

    # 求 texture -> image 的单应（4 点 DLT），PIL 需要 target->source 的逆映射
    H = _homography(src, dst)
    H_inv = np.linalg.inv(H)
    warped = texture.transform((w, h), Image.PERSPECTIVE, tuple(H_inv.flatten()[:8]), resample=Image.BICUBIC)

    # 合成到深灰背景 + 轻微噪声与模糊，模拟真实照片
    bg = Image.new("RGB", (w, h), (35, 38, 45))
    mask = _quad_mask((w, h), dst)
    bg.paste(warped, (0, 0), mask)

    arr = np.asarray(bg, dtype=np.float32)
    rng = np.random.default_rng(abs(hash((cam_pos.tobytes()))) % 2**31)
    arr += rng.normal(0, 2.5, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
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


def build_dataset(out_dir: Path, views: int, size: int, plane: float = 3.0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()
    texture = make_texture(768)
    radius = plane * 1.2
    focal = size * 0.9
    written = 0
    for i in range(views):
        theta = 2 * math.pi * i / views
        # 环绕 + 高度起伏：既有水平基线，也有俯仰变化（更接近手持环绕拍摄）
        cam = np.array([
            radius * math.cos(theta),
            0.45 * math.sin(2 * theta) + 0.15,
            radius * math.sin(theta),
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
    n = build_dataset(out_dir, args.views, args.size)
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
