"""从 `model_gauss.glb` 反解高斯，并用 gsplat 渲染，与"直接读 PLY"的渲染逐像素比对。

目的（回答"完整的 glb 到底有没有保住质量"）：
* 解析 glb 的私有属性（`_GS_SCALE` / `_GS_ROTATION` / `_GS_OPACITY` / `_GS_SH_00..11`）；
* 还原成 3DGS 参数（SH 由 DC + 48 残差拼回，尺度取 exp、不透明度取 sigmoid 前的原始值口径需对齐）；
* 用同一台相机分别渲染 glb 还原的模型与原始 PLY 模型；
* 报告两者的 PSNR（应 >50 dB，即几乎逐像素相同）。

用法：
  python scripts/render_from_glb.py --job img5-solid-v3 [--view 0]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from gsplat import rasterization  # noqa: E402

from gsflow.gsdata.colmap_io import read_model  # noqa: E402
from gsflow.gsdata.mesh_export import scale_intrinsics  # noqa: E402
from gsflow.gsdata.shio import read_gaussians_full  # noqa: E402
from gsflow.paths import apply_runtime_env  # noqa: E402

# ⚠ 必须先把 torch JIT 缓存指向工作区（否则会用用户目录、被沙箱拒绝后重建失败）
apply_runtime_env()


def read_glb(path: Path):
    raw = path.read_bytes()
    magic, ver, _ = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67, "不是 glb"
    jlen, jtype = struct.unpack("<II", raw[12:20])
    gltf = json.loads(raw[20:20 + jlen].decode("utf-8"))
    off = 20 + jlen
    blen, btype = struct.unpack("<II", raw[off:off + 8])
    return gltf, raw[off + 8:off + 8 + blen]


def accessor(gltf, blob, idx):
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    ncomp = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc["type"]]
    dtype = {5126: "<f4", 5121: "u1"}[acc["componentType"]]
    arr = np.frombuffer(blob, dtype=dtype, count=acc["count"] * ncomp,
                        offset=bv["byteOffset"]).reshape(acc["count"], ncomp)
    return arr.astype(np.float32)


def gauss_from_glb(path: Path) -> dict:
    gltf, blob = read_glb(path)
    ext = gltf["extensions"]["GSFLOW_gaussian_splatting"]
    attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
    n = int(ext["gaussianCount"])
    pos = accessor(gltf, blob, attrs["POSITION"]).copy()
    pos[:, 1] *= -1.0
    pos[:, 2] *= -1.0
    scale = accessor(gltf, blob, attrs["_GS_SCALE"])
    rot = accessor(gltf, blob, attrs["_GS_ROTATION"])
    opac = accessor(gltf, blob, attrs["_GS_OPACITY"]).reshape(n, 1)
    n_vec4 = int(ext["shRestVec4"])
    rest = np.concatenate([accessor(gltf, blob, attrs[f"_GS_SH_{i:02d}"]) for i in range(n_vec4)],
                          axis=1)
    d = (ext.get("shDegree") or 0)
    k_rest = ((d + 1) ** 2) - 1
    rest = rest[:, : k_rest * 3].reshape(n, k_rest, 3)
    if "_GS_SH_DC" in attrs:
        dc_sh = accessor(gltf, blob, attrs["_GS_SH_DC"])          # 无损
    else:
        dc = accessor(gltf, blob, attrs["COLOR_0"]).astype(np.float32) / 255.0
        dc_sh = (dc - 0.5) / 0.28209479177387814                 # 8 位近似（旧文件）
    shs = np.concatenate([dc_sh[:, None, :], rest], axis=1)
    return {"xyz": pos, "scales": scale, "quats": rot, "opacity": opac.reshape(-1),
            "shs": shs, "n": n, "sh_degree": d}


def render(g: dict, c2w, K, W, H) -> np.ndarray:
    dev = "cuda"
    means = torch.as_tensor(g["xyz"], dtype=torch.float32, device=dev)
    quats = torch.as_tensor(g["quats"], dtype=torch.float32, device=dev)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.as_tensor(g["scales"], dtype=torch.float32, device=dev)
    opac = torch.as_tensor(g["opacity"], dtype=torch.float32, device=dev).clamp(0, 1)
    shs = torch.as_tensor(g["shs"], dtype=torch.float32, device=dev)
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    with torch.no_grad():
        colors, alphas, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
            viewmats=torch.as_tensor(w2c[None], dtype=torch.float32, device=dev),
            Ks=torch.as_tensor(K[None], dtype=torch.float32, device=dev),
            width=W, height=H, sh_degree=int(g["sh_degree"]), near_plane=0.01,
            far_plane=1e10, packed=False, render_mode="RGB",
        )
        img = colors[0] + (1 - alphas[0])
    return img.clamp(0, 1).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--glb", default="model_gauss.glb")
    ap.add_argument("--view", type=int, default=0)
    args = ap.parse_args()

    job = ROOT / "workspace" / args.job
    data = job / "04_train" / "data"
    sp = data / "sparse"
    rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
    imgs = sorted(rec.images, key=lambda im: im.name)
    im = imgs[args.view]
    cam = rec.cameras[im.camera_id]
    W, H = cam.width, cam.height
    K = scale_intrinsics(np.asarray(cam.intrinsics(), dtype=np.float64), (W, H), (W, H))
    R, ctr = im.cam_to_world()

    # ⚠ 必须用导出时实际用的那个检查点：
    # 各 checkpoint 的高斯数可能完全相同（405,629），文件大小也相同，
    # 按"文件最大"挑会拿到任意一个（实测拿到了 iteration_10000）。
    manifest = job / "05_export" / "manifest.json"
    ply = None
    if manifest.exists():
        try:
            src = json.loads(manifest.read_text(encoding="utf-8")).get("source_ply")
            if src and Path(src).exists():
                ply = Path(src)
        except Exception:
            pass
    if ply is None:
        ply = sorted((job / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
                     key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))[-1]
    print(f"  对比用检查点: {ply.parent.name}")
    g_ply = read_gaussians_full(ply)
    g_glb = gauss_from_glb(job / "05_export" / args.glb)
    print(f"  PLY 高斯 {g_ply['n']:,}  |  glb 高斯 {g_glb['n']:,}  |  SH 阶 {g_glb['sh_degree']}")
    # 直接比参数：比渲染更容易定位差异
    for key in ("xyz", "scales", "quats", "opacity", "shs"):
        a_ = np.asarray(g_ply[key], dtype=np.float64).reshape(len(g_ply["xyz"]), -1)
        b_ = np.asarray(g_glb[key], dtype=np.float64).reshape(len(g_glb["xyz"]), -1)
        if a_.shape != b_.shape:
            print(f"   {key}: 形状不同 {a_.shape} vs {b_.shape}")
            continue
        d_ = np.abs(a_ - b_)
        print(f"   {key:8s} 最大差 {d_.max():.3e}  均值 {d_.mean():.3e}")

    a = render(g_ply, np.eye(4) @ np.block([[np.asarray(R), np.asarray(ctr)[:, None]], [np.zeros(3), 1]]),
               K, W, H)
    b = render(g_glb, np.eye(4) @ np.block([[np.asarray(R), np.asarray(ctr)[:, None]], [np.zeros(3), 1]]),
               K, W, H)
    mse = float(((a - b) ** 2).mean())
    psnr = 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)
    print(f"  渲染对比：MSE={mse:.3e}  PSNR={psnr:.2f} dB  （>45 dB 即可认为无损）")
    print(f"  最大绝对差：{float(np.abs(a - b).max()):.5f}")

    from PIL import Image
    out = ROOT / "env" / "tmp" / "glb_vs_ply.png"
    side = np.concatenate([a, b], axis=1)
    Image.fromarray((side * 255).astype(np.uint8)).save(out)
    print(f"  已写对比图 {out}")
    return 0 if psnr > 45 else 1


if __name__ == "__main__":
    raise SystemExit(main())
