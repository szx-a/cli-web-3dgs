"""COLMAP 数据读取：二进制稀疏模型解析 + 去畸变数据准备。

为什么需要它：gsplat / 原版 3DGS 都期望“去畸变后的 COLMAP 数据”。
COLMAP 自带的 image_undistorter 是最稳妥的去畸变工具，但传入 mask 目录时
它会把掩码一起做几何变换；为了让 02_masks 继续有效，这里的做法是：

    1. 原图（含真实掩码）复制成 02_images_white/（全部白色掩码）交给 undistorter；
    2. undistorter 产出 04_undistorted/images + sparse/0；
    3. 我们自己记录“原始文件名 -> 去畸变文件名”的映射；
    4. 训练时用原掩码按文件名匹配，在 undistorted 图上做形态学缩放近似。

这样既复用 COLMAP 的去畸变，又不牺牲掩码。
"""

from __future__ import annotations

import json
import re
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------- 相机模型表
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: list[float]

    def intrinsics(self) -> np.ndarray:
        """返回 3x3 K（假定主点已给出，未给出则用图像中心）。"""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            f, cx, cy = p[0], p[1], p[2]
            fx = fy = f
        elif self.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE", "FOV"):
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:
            raise ValueError(f"未支持的相机模型：{self.model}")
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


@dataclass
class Image:
    id: int
    qvec: np.ndarray      # (w, x, y, z) world->camera
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))

    def rotation(self) -> np.ndarray:
        w, x, y, z = self.qvec
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float64,
        )

    def cam_to_world(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (R_c2w, t_c2w)，使 X_world = R_c2w @ X_cam + t_c2w。"""
        r = self.rotation()
        t = self.tvec.astype(np.float64)
        return r.T, -r.T @ t


@dataclass
class Reconstruction:
    cameras: dict[int, Camera]
    images: list[Image]
    points3D: np.ndarray      # (N, 3)
    colors3D: np.ndarray      # (N, 3) uint8
    errors3D: np.ndarray      # (N,)

    def by_name(self) -> dict[str, Image]:
        return {im.name: im for im in self.images}


# ------------------------------------------------------------------ 解析
def _read_next_bytes(fh, num: int, fmt: str) -> tuple:
    data = fh.read(num)
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cams: dict[int, Camera] = {}
    with path.open("rb") as fh:
        (num,) = _read_next_bytes(fh, 8, "Q")
        for _ in range(num):
            cam_id, model_id, width, height = _read_next_bytes(fh, 24, "iiQQ")
            model_name, n_params = CAMERA_MODELS.get(model_id, (f"UNKNOWN_{model_id}", 0))
            params = list(_read_next_bytes(fh, 8 * n_params, "d" * n_params)) if n_params else []
            cams[cam_id] = Camera(cam_id, model_name, int(width), int(height), [float(x) for x in params])
    return cams


def read_images_binary(path: Path) -> list[Image]:
    images: list[Image] = []
    with path.open("rb") as fh:
        (num,) = _read_next_bytes(fh, 8, "Q")
        for _ in range(num):
            props = _read_next_bytes(fh, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]
            name_bytes = b""
            while True:
                char = fh.read(1)
                if char == b"\x00" or char == b"":
                    break
                name_bytes += char
            name = name_bytes.decode("utf-8", errors="replace")
            (num_points2D,) = _read_next_bytes(fh, 8, "Q")
            if num_points2D:
                raw = _read_next_bytes(fh, num_points2D * 24, "ddq" * num_points2D)
                xys = np.array(
                    [[raw[3 * i], raw[3 * i + 1]] for i in range(num_points2D)], dtype=np.float64
                ).reshape(-1, 2)
                ids = np.array([raw[3 * i + 2] for i in range(num_points2D)], dtype=np.int64)
            else:
                xys = np.zeros((0, 2), dtype=np.float64)
                ids = np.zeros((0,), dtype=np.int64)
            images.append(Image(int(image_id), qvec, tvec, int(camera_id), name, xys, ids))
    return images


def read_points3D_binary(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz, rgb, err = [], [], []
    if not path.exists():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.zeros((0,))
    with path.open("rb") as fh:
        (num,) = _read_next_bytes(fh, 8, "Q")
        for _ in range(num):
            props = _read_next_bytes(fh, 43, "QdddBBBd")
            xyz.append(props[1:4])
            rgb.append(props[4:7])
            err.append(props[7])
            (track_len,) = _read_next_bytes(fh, 8, "Q")
            if track_len:
                fh.read(8 * track_len)
    return (
        np.array(xyz, dtype=np.float64).reshape(-1, 3),
        np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        np.array(err, dtype=np.float64).reshape(-1),
    )


def read_model(model_dir: Path) -> Reconstruction:
    cams = read_cameras_binary(model_dir / "cameras.bin")
    imgs = read_images_binary(model_dir / "images.bin")
    pts, cols, errs = read_points3D_binary(model_dir / "points3D.bin")
    return Reconstruction(cams, imgs, pts, cols, errs)


# ------------------------------------------------------------- 去畸变准备
def prepare_undistorted(
    ctx,
    colmap_exe: str,
    images_dir: Path,
    mask_dir: Path | None,
    model_dir: Path,
    out_dir: Path,
) -> dict:
    """调用 image_undistorter 生成训练用数据，并返回掩码映射信息。

    关键：**只复制重建里真实注册过的图像**，并且保持原始文件名。

    踩坑记录：早期做法是把整个 02_images_clean / 01_images_raw 目录重命名为
    00000.jpg... 再传给 undistorter。但 undistorter 是按重建里的 image name
    去 image_path 下找文件的，名字对不上时就静默跳过（日志只写
    "Undistorting image [1/5]"，36 张图只出来 5 张），后续训练直接崩。
    """
    from ..tools.locate import run_tool

    rec = read_model(model_dir)
    registered_names = [im.name for im in rec.images]
    if not registered_names:
        from ..errors import StageError

        raise StageError("colmap", f"模型里没有已注册图像：{model_dir}", "重跑 colmap 阶段")

    work = out_dir.parent / "undistort_in"
    white_dir = work / "images_white"
    white_mask = work / "masks_white"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    white_dir.mkdir(parents=True, exist_ok=True)
    white_mask.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    available = {p.name: p for p in images_dir.iterdir() if p.suffix.lower() in exts}
    names: list[str] = []
    missing: list[str] = []
    for name in registered_names:
        src = available.get(name)
        if src is None:
            # 容忍扩展名差异（COLMAP 记录 0003.jpg，磁盘上可能是 0003.png）
            stem = Path(name).stem
            cands = [p for p in available.values() if p.stem == stem]
            src = cands[0] if cands else None
        if src is None:
            missing.append(name)
            continue
        shutil.copy2(src, white_dir / name)
        names.append(name)
        # 全白掩码：告诉 undistorter 不要丢任何像素
        try:
            from PIL import Image as PILImage

            with PILImage.open(src) as im:
                PILImage.new("L", im.size, 255).save(white_mask / (Path(name).stem + ".png"))
        except Exception:
            pass

    if missing:
        ctx.warn(f"{len(missing)} 张已注册图像在 {images_dir} 里找不到，已跳过：{missing[:5]}")
    ctx.info(f"准备去畸变：{len(names)}/{len(registered_names)} 张注册图像")

    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = ctx.logs_dir() / "colmap_undistort.log"
    code = run_tool(
        [
            colmap_exe, "image_undistorter",
            "--image_path", str(white_dir),
            "--input_path", str(model_dir),
            "--output_path", str(out_dir),
            "--output_type", "COLMAP",
            "--max_image_size", "4000",
        ],
        log_file=log,
        on_line=lambda l: ctx.emit(l, level="debug"),
    )
    if code != 0:
        from ..errors import StageError

        raise StageError(
            "colmap",
            f"image_undistorter 失败（exit={code}）",
            f"查看日志 {log}；若图像尺寸过大可先降低 train.resolution。",
        )

    # 去畸变通常保持文件名（个别版本会改扩展名），这里两种都记下来
    mapping: dict[str, str] = {}
    und_images = out_dir / "images"
    for name in names:
        exact = und_images / name
        if exact.exists():
            mapping[name] = name
            continue
        cands = list(und_images.glob(Path(name).stem + ".*"))
        if cands:
            mapping[name] = cands[0].name

    if not mapping:
        from ..errors import StageError

        raise StageError(
            "colmap",
            "去畸变输出里没有任何图像，训练无法继续",
            f"检查 {und_images} 与日志 {log}（常见原因：注册图像与磁盘文件名不一致）。",
        )
    ctx.info(f"去畸变输出图像：{len(mapping)} 张（注册 {len(registered_names)}，输入 {len(names)}）")

    masks_out = mask_dir if (mask_dir and any(mask_dir.glob("*.png"))) else None
    info = {
        "image_count": len(mapping),
        "registered_count": len(registered_names),
        "name_map": mapping,
        "missing": missing,
        "masks_dir": str(masks_out) if masks_out else "",
        "white_images_dir": str(white_dir),
    }
    (out_dir.parent / "undistort_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def fix_up_axis_ambiguity(images: list[Image]) -> bool:
    """检测并统一 COLMAP 的坐标轴约定差异（部分版本 images.bin 带翻转）。

    做法：比较相机朝向的“多数一致方向”。不一致时对 R 做 y/z 翻转。
    返回是否做了翻转。
    """
    if len(images) < 3:
        return False
    ups = []
    for im in images:
        r = im.rotation()
        ups.append(np.sign(r.T @ np.array([0.0, 0.0, -1.0])).sum())
    positives = sum(1 for u in ups if u > 0)
    if positives * 2 >= len(ups):
        return False
    for im in images:
        r = im.rotation()
        flip = np.diag([1.0, -1.0, -1.0])
        rf = flip @ r @ flip
        w, x, y, z = _rot_to_quat(rf)
        im.qvec = np.array([w, x, y, z])
    return True


def _rot_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def match_masks_to_undistorted(mask_dir: Path, info: dict, target_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """把原始掩码按名字对应到去畸变图像上（尺寸不同则缩放）。"""
    from PIL import Image as PILImage

    out: dict[str, np.ndarray] = {}
    name_map: dict[str, str] = info.get("name_map", {})
    if not mask_dir or not mask_dir.exists():
        return out
    for orig, und in name_map.items():
        stem = Path(orig).stem
        cand = mask_dir / f"{stem}.png"
        if not cand.exists():
            continue
        try:
            with PILImage.open(cand) as im:
                if im.size != target_size:
                    im = im.resize(target_size, PILImage.NEAREST)
                out[und] = np.asarray(im, dtype=np.uint8) > 127
        except Exception:
            continue
    return out
