"""阶段 5：模型查看与导出。

产物：
  05_export/model.glb        点云式 glTF（通用查看器/游戏引擎可直接打开）
  05_export/model.spz        Niantic SPZ 压缩格式（体积约为 PLY 的 1/10，three.js 生态通用）
  05_export/viewer.html      离线可用的本地查看器（内嵌模型，双击即看）
  05_export/preview_*.png    若干视角的渲染预览图
  05_export/manifest.json    产物清单

说明：3DGS 的 PLY 不是带面的网格，所以这里的 glb 用 glTF 的 POINTS 图元承载
每个高斯的颜色，兼容性最好；真正带三角面的网格化（SuGaR/2DGS 路线）属于后续增强。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from ..errors import StageError
from ..pipeline.base import Artifact, Stage, StageResult
from ..pipeline.context import StageContext

SH_C0 = 0.28209479177387814
GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
COMPONENT_FLOAT = 5126
COMPONENT_UINT = 5125
MODE_POINTS = 0
MODE_TRIANGLES = 4


def validate_glb(path: Path) -> dict:
    """按 glTF 2.0 规范校验我们导出的 GLB，返回 {ok, problems, info}。

    为什么要单独做：GLB 是二进制容器，"文件有内容"并不代表能被查看器读出来。
    这里检查 magic/版本/总长、chunk 结构、bufferView 的 4 字节对齐与越界、
    POSITION 的 min/max、以及图元模式是否为 POINTS。
    """
    problems: list[str] = []
    info: dict = {}
    raw = path.read_bytes()
    if len(raw) < 20:
        return {"ok": False, "problems": ["文件过小，不是合法 GLB"], "info": {}}

    magic, version, total = struct.unpack("<III", raw[:12])
    if magic != GLB_MAGIC:
        problems.append("magic 不是 glTF")
    if version != 2:
        problems.append(f"glTF 版本不是 2（{version}）")
    if total != len(raw):
        problems.append(f"头部声明长度 {total} != 实际 {len(raw)}")

    off = 12
    chunks: list[tuple[int, int, int]] = []
    while off + 8 <= len(raw):
        clen, ctype = struct.unpack("<II", raw[off:off + 8])
        chunks.append((ctype, clen, off + 8))
        off += 8 + clen
    kinds = [c[0] for c in chunks]
    if JSON_CHUNK not in kinds:
        problems.append("缺少 JSON chunk")
    if BIN_CHUNK not in kinds:
        problems.append("缺少 BIN chunk")
    if problems:
        return {"ok": False, "problems": problems, "info": info}

    j_idx, b_idx = kinds.index(JSON_CHUNK), kinds.index(BIN_CHUNK)
    try:
        gltf = json.loads(raw[chunks[j_idx][2]:chunks[j_idx][2] + chunks[j_idx][1]].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"ok": False, "problems": [f"JSON chunk 解析失败：{exc}"], "info": {}}

    bin_len = chunks[b_idx][1]
    for i, bv in enumerate(gltf.get("bufferViews", [])):
        if bv["byteOffset"] % 4 != 0:
            problems.append(f"bufferView[{i}] 起始地址未 4 字节对齐")
        if bv["byteOffset"] + bv["byteLength"] > bin_len:
            problems.append(f"bufferView[{i}] 越界（{bv['byteOffset']}+{bv['byteLength']} > {bin_len}）")

    try:
        prim = gltf["meshes"][0]["primitives"][0]
        pos = gltf["accessors"][prim["attributes"]["POSITION"]]
        col = gltf["accessors"][prim["attributes"]["COLOR_0"]]
    except (KeyError, IndexError) as exc:
        return {"ok": False, "problems": [f"缺少必要的 mesh/accessor：{exc}"], "info": {}}

    if not pos.get("min") or not pos.get("max"):
        problems.append("POSITION accessor 缺少 min/max（glTF 规范要求）")
    if prim.get("mode", 4) != MODE_POINTS:
        problems.append(f"图元模式不是 POINTS（{prim.get('mode')}）")
    if pos["count"] != col["count"]:
        problems.append("POSITION 与 COLOR_0 顶点数不一致")

    info = {
        "bytes": len(raw),
        "chunks": len(chunks),
        "mode": prim.get("mode", 4),
        "count": pos["count"],
        "materials": len(gltf.get("materials", [])),
        "extensions": gltf.get("extensionsUsed", []),
    }
    return {"ok": not problems, "problems": problems, "info": info}


class ExportStage(Stage):
    name = "export"
    title = "⑤ 导出与预览"
    description = "把 point_cloud.ply 转成 glb / spz，并生成离线查看器与预览图"
    requires = ("train",)
    optional = True
    inputs = (Artifact("04_train/point_cloud", "训练产物目录"),)
    outputs = (Artifact("05_export/manifest.json", "导出清单"),)

    def run(self, ctx: StageContext) -> StageResult:
        ply = self._find_ply(ctx.job_dir)
        if ply is None:
            raise StageError(self.name, "找不到 point_cloud.ply", "先完成 train 阶段，或在作业里指定 --set export.ply=<路径>")
        ctx.info(f"模型文件：{ply} ({ply.stat().st_size/1024/1024:.1f} MB)")

        out_dir = ctx.stage_dir("05_export")
        # 每次重跑导出都清空旧产物：否则 manifest.json 会保留上一次的记录，
        # 把已经改过单位的旧条目一起带出来（我们踩过这个）。
        for old in out_dir.iterdir():
            if old.is_file():
                old.unlink()
        gauss = self._read_ply(ply, max_points=int(ctx.cfg("glb_max_points", 800000) or 0))
        ctx.info(f"读取到 {len(gauss['xyz'])} 个高斯")

        formats = ctx.cfg("formats", ["glb_points", "viewer_html"])
        if isinstance(formats, str):
            formats = [f.strip() for f in formats.strip("[]").split(",") if f.strip()]
        produced: list[dict] = []

        if "glb_points" in formats or "glb" in formats:
            glb_path = out_dir / "model.glb"
            shrink = float(ctx.cfg("glb_shrink", 1.0) or 1.0)
            self._write_glb(glb_path, gauss, shrink=shrink)
            produced.append({"kind": "glb", "path": glb_path.name,
                             "size_mb": round(glb_path.stat().st_size / 1024 / 1024, 2),
                             "desc": "点云式 glTF，通用查看器可直接打开"})
            ctx.success(f"已导出 {glb_path.name}（{produced[-1]['size_mb']} MB）")

        if "spz" in formats or "splat" in formats:
            try:
                spz_path = self._write_spz(out_dir / "model.spz", gauss)
                if spz_path:
                    produced.append({"kind": "spz", "path": spz_path.name,
                                     "size_mb": round(spz_path.stat().st_size / 1024 / 1024, 2),
                                     "desc": "SPZ 压缩格式（three.js / SuperSplat 生态）"})
                    ctx.success(f"已导出 {spz_path.name}")
            except Exception as exc:
                ctx.warn(f"SPZ 导出失败（不影响其它产物）：{exc}")

        if "obj" in formats:
            obj_path = out_dir / "model.obj"
            self._write_obj(obj_path, gauss)
            produced.append({"kind": "obj", "path": obj_path.name,
                             "size_mb": round(obj_path.stat().st_size / 1024 / 1024, 2),
                             "desc": "通用点云 OBJ"})

        previews = self._render_previews(ctx, gauss, out_dir)
        produced.extend(previews)

        if "viewer_html" in formats:
            html = out_dir / "viewer.html"
            self._write_viewer(html, gauss, ctx)
            produced.append({"kind": "html", "path": html.name, "size_mb": round(html.stat().st_size / 1024 / 1024, 2),
                             "desc": "离线查看器（双击打开）"})
            ctx.success(f"已生成 {html.name}")

        manifest = {
            "source_ply": str(ply),
            "gaussians": int(len(gauss["xyz"])),
            "formats": formats,
            "artifacts": produced,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return StageResult(artifacts=[str(out_dir / "manifest.json")], metrics=manifest,
                           message=f"导出 {len(produced)} 个产物")

    # ------------------------------------------------------------------ 读取
    @staticmethod
    def _find_ply(job_dir: Path) -> Path | None:
        """找迭代次数最高的 point_cloud.ply。

        踩坑：不能直接按路径字符串排序——`iteration_5000` 会排在
        `iteration_30000` 之后（字典序），导致导出的是早期检查点。
        这里优先读训练阶段写在 metrics.json 里的最终模型路径。
        """
        import re

        metrics = job_dir / "04_train" / "metrics.json"
        if metrics.exists():
            try:
                data = json.loads(metrics.read_text(encoding="utf-8"))
                recorded = Path(str(data.get("ply", "")))
                if recorded.exists():
                    return recorded
            except (json.JSONDecodeError, OSError, TypeError):
                pass

        base = job_dir / "04_train" / "point_cloud"
        if not base.exists():
            return None
        cands = list(base.rglob("point_cloud.ply"))
        if not cands:
            return None

        def iteration(p: Path) -> int:
            m = re.search(r"iteration_(\d+)", str(p))
            return int(m.group(1)) if m else -1

        return max(cands, key=iteration)

    @staticmethod
    def _read_ply(ply: Path, max_points: int = 0) -> dict:
        """读取 3DGS PLY，把 SH DC 分量解成 RGB；可选按不透明度截断。"""
        from plyfile import PlyData

        data = PlyData.read(str(ply))
        v = data["vertex"]
        names = set(v.data.dtype.names or ())
        xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)

        if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
            dc = np.stack([np.asarray(v["f_dc_0"]), np.asarray(v["f_dc_1"]), np.asarray(v["f_dc_2"])], axis=1)
            rgb = np.clip(dc * SH_C0 + 0.5, 0.0, 1.0) * 255.0
        elif {"red", "green", "blue"} <= names:
            rgb = np.stack([np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])], axis=1).astype(np.float32)
        else:
            rgb = np.full_like(xyz, 128.0)

        if "opacity" in names:
            op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"]).astype(np.float32)))
        else:
            op = np.ones(len(xyz), dtype=np.float32)

        if {"scale_0", "scale_1", "scale_2"} <= names:
            scales = np.exp(np.stack([np.asarray(v["scale_0"]), np.asarray(v["scale_1"]),
                                      np.asarray(v["scale_2"])], axis=1).astype(np.float32))
        else:
            scales = np.full_like(xyz, 0.01)

        if {"rot_0", "rot_1", "rot_2", "rot_3"} <= names:
            quats = np.stack([np.asarray(v["rot_0"]), np.asarray(v["rot_1"]),
                              np.asarray(v["rot_2"]), np.asarray(v["rot_3"])], axis=1).astype(np.float32)
        else:
            quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(xyz), 1))

        order = np.argsort(-op)
        if max_points and len(order) > max_points:
            order = order[:max_points]
        return {
            "xyz": xyz[order], "rgb": rgb[order], "opacity": op[order],
            "scales": scales[order], "quats": quats[order],
        }

    # ------------------------------------------------------------------ GLB
    def _write_glb(self, path: Path, gauss: dict, shrink: float = 1.0) -> Path:
        """写最小可用的 glTF 2.0（POINTS 图元 + 顶点色 + 可选缩放）。"""
        xyz = gauss["xyz"] * float(shrink)
        rgb_u8 = np.clip(gauss["rgb"], 0, 255).astype(np.uint8)
        # glTF 约定：Y 轴向上，COLMAP 是 Y 向下，这里翻转 y/z 让模型正立
        xyz = xyz.copy()
        xyz[:, 1] *= -1
        xyz[:, 2] *= -1

        positions = xyz.astype(np.float32).tobytes()
        colors = rgb_u8.tobytes()
        # 颜色按 4 字节对齐
        pad_c = (-len(colors)) % 4
        colors_padded = colors + b"\x00" * pad_c
        bin_blob = positions + colors_padded
        pos_len, col_len = len(positions), len(colors_padded)

        gltf = {
            "asset": {"version": "2.0", "generator": "gsflow"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "gaussians"}],
            "meshes": [{
                "name": "gaussians",
                "primitives": [{
                    "attributes": {"POSITION": 0, "COLOR_0": 1},
                    "mode": MODE_POINTS,
                    "material": 0,
                }],
            }],
            "materials": [{
                "name": "splat",
                "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.6},
                "extensions": {"KHR_materials_unlit": {}},
            }],
            "extensionsUsed": ["KHR_materials_unlit"],
            "buffers": [{"byteLength": len(bin_blob)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": pos_len, "target": 34962},
                {"buffer": 0, "byteOffset": pos_len, "byteLength": col_len, "target": 34962},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": COMPONENT_FLOAT, "count": len(xyz), "type": "VEC3",
                 "min": [float(xyz[:, i].min()) for i in range(3)],
                 "max": [float(xyz[:, i].max()) for i in range(3)]},
                {"bufferView": 1, "componentType": 5121, "normalized": True, "count": len(xyz), "type": "VEC4"
                 if False else "VEC3"},
            ],
            "extras": {"gaussian_count": int(len(xyz)), "source": "3dgs-ply"},
        }
        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_pad = (-len(json_bytes)) % 4
        json_bytes += b" " * json_pad

        total = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
        with path.open("wb") as fh:
            fh.write(struct.pack("<III", GLB_MAGIC, 2, total))
            fh.write(struct.pack("<II", len(json_bytes), JSON_CHUNK))
            fh.write(json_bytes)
            fh.write(struct.pack("<II", len(bin_blob), BIN_CHUNK))
            fh.write(bin_blob)
        return path

    # ------------------------------------------------------------------ SPZ
    def _write_spz(self, path: Path, gauss: dict) -> Path | None:
        """调用 spz 官方 python 包（若已安装）；未安装则跳过。"""
        try:
            from spz import GaussianSplat  # type: ignore
        except ImportError:
            return None
        gs = GaussianSplat(
            positions=gauss["xyz"], scales=gauss["scales"], rotations=gauss["quats"],
            alphas=gauss["opacity"], colors=gauss["rgb"] / 255.0,
        )
        path.write_bytes(bytes(gs.to_spz()))
        return path

    # ------------------------------------------------------------------ OBJ
    def _write_obj(self, path: Path, gauss: dict) -> Path:
        xyz = gauss["xyz"]
        rgb = np.clip(gauss["rgb"], 0, 255).astype(int)
        with path.open("w", encoding="utf-8") as fh:
            fh.write("# gsflow point cloud export\n")
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                fh.write(f"v {x:.6f} {-y:.6f} {-z:.6f} {r/255:.4f} {g/255:.4f} {b/255:.4f}\n")
        return path

    # ------------------------------------------------------------------ 预览
    def _render_previews(self, ctx: StageContext, gauss: dict, out_dir: Path, count: int = 4) -> list[dict]:
        """生成预览图：优先用 **gsplat 真实渲染**，失败时退回 matplotlib 散点投影。

        为什么要真渲染：matplotlib 版只是把高斯**中心点**画成散点，
        得到的是一团点云，看不出重建质量（一个 PSNR 35dB 的好模型和一堆乱点长得一样）。
        真渲染能直接反映"模型看到的世界"，也是用户最关心的产物。

        gsplat 不可用时（例如无 GPU 环境）自动退回散点图，保证导出阶段仍能独立跑通。
        """
        rendered = self._render_previews_splat(ctx, gauss, out_dir, count)
        if rendered:
            return rendered
        return self._render_previews_scatter(ctx, gauss, out_dir, count)

    def _render_previews_splat(self, ctx: StageContext, gauss: dict, out_dir: Path,
                               count: int = 4) -> list[dict]:
        """用 gsplat 从若干环绕视角真实渲染模型（无头，不需要 GUI）。"""
        try:
            import torch
            from gsplat import rasterization
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            ctx.info(f"未启用真实渲染预览（{type(exc).__name__}），改用散点投影")
            return []

        try:
            if not torch.cuda.is_available():
                ctx.info("无 CUDA 设备，预览改用散点投影")
                return []
            xyz = torch.as_tensor(gauss["xyz"], dtype=torch.float32, device="cuda")
            n = xyz.shape[0]
            if n == 0:
                return []
            # 注意：_read_ply 已经把 PLY 里的 log 尺度**指数化**过了，
            # 这里拿到的是真实尺度，绝对不能再 exp 一次
            # （早期在预览里又 exp 了一遍 -> 尺度变成天文数字 -> 渲染一片空白）。
            scales_real = torch.as_tensor(gauss["scales"], dtype=torch.float32, device="cuda")
            quats = torch.as_tensor(gauss["quats"], dtype=torch.float32, device="cuda")
            rgb = torch.as_tensor(gauss["rgb"], dtype=torch.float32, device="cuda") / 255.0

            opac0 = torch.as_tensor(gauss["opacity"], dtype=torch.float32, device="cuda").reshape(-1)
            # 只保留"看得见"的高斯：低不透明度的浮点数会把取景撑大、把画面渲成一片背景色。
            keep = opac0 > 0.02
            if int(keep.sum()) < 16:
                keep = torch.ones_like(opac0, dtype=torch.bool)
            xyz = xyz[keep]
            scales_real = scales_real[keep]
            quats = quats[keep]
            opac = opac0[keep]
            rgb = rgb[keep]
            n = int(xyz.shape[0])
            ctx.info(f"预览渲染使用 {n} 个可见高斯（不透明度 > 0.02）")

            # 首选：直接用**训练时的真实相机位姿**渲染。
            # 自己估的"环绕相机"很容易取景失当（实测把画面渲成一片背景色），
            # 而训练相机的视角天然合理，也正好能反映"模型在训练视图上看到什么"。
            poses = self._training_cameras(ctx, out_dir)
            if poses:
                return self._render_from_poses(ctx, xyz, scales_real, quats, opac, rgb, poses, out_dir, count)

            center = xyz.mean(dim=0)
            spread = torch.linalg.norm(xyz - center, dim=1) + scales_real.max(dim=1).values
            radius = float(spread.max()) * 2.6 + 1e-3
            H = W = 512
            fov = 1.0
            focal = 0.5 * W / np.tan(0.5 * fov)
            shs = ((rgb - 0.5) / 0.28209479177387814).reshape(n, 1, 3)

            angles = [("front", 0.0, 0.15), ("side", 90.0, 0.15),
                      ("top", 0.0, 1.35), ("iso", 45.0, 0.55)]
            out: list[dict] = []
            for name, az_deg, el in angles[:count]:
                az = np.radians(az_deg)
                eye = center + torch.tensor(
                    [radius * np.cos(el) * np.cos(az),
                     radius * np.cos(el) * np.sin(az),
                     radius * np.sin(el)], dtype=torch.float32, device="cuda")
                fwd = center - eye
                fwd = fwd / (fwd.norm() + 1e-9)
                up_w = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
                right = torch.cross(up_w, fwd, dim=0)
                if float(right.norm()) < 1e-6:
                    right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
                right = right / (right.norm() + 1e-9)
                up = torch.cross(fwd, right, dim=0)
                up = up / (up.norm() + 1e-9)
                R = torch.stack([right, up, fwd], dim=0)
                t = -R @ eye
                viewmat = torch.eye(4, dtype=torch.float32, device="cuda")
                viewmat[:3, :3] = R
                viewmat[:3, 3] = t
                K = torch.tensor([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1.0]],
                                 dtype=torch.float32, device="cuda")
                with torch.no_grad():
                    colors, alphas, _ = rasterization(
                        means=xyz, quats=quats, scales=scales_real,
                        opacities=opac.clamp(0, 1), colors=shs,
                        viewmats=viewmat[None], Ks=K[None],
                        width=W, height=H, sh_degree=0, packed=False, render_mode="RGB",
                    )
                    img = colors[0] + (1 - alphas[0])
                arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                p = out_dir / f"preview_{name}.png"
                Image.fromarray(arr).save(p)
                kb = round(p.stat().st_size / 1024, 1)
                out.append({"kind": "preview", "path": p.name, "size_kb": kb, "size_mb": round(kb / 1024, 3),
                            "desc": f"{name} 视角渲染（gsplat 真实渲染）"})
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"真实渲染预览失败（{type(exc).__name__}: {exc}），改用散点投影")
            return []
        if out:
            ctx.info(f"已生成 {len(out)} 张真实渲染预览（gsplat）")
        return out

    def _training_cameras(self, ctx: StageContext, out_dir: Path):
        """读取训练阶段用的 COLMAP 稀疏模型，取若干真实相机位姿（c2w）与 K。

        为什么不用"自己估的环绕相机"：实测那样很容易取景失当
        （相机落在高斯云内部或太远），渲出来是一片背景色，
        而训练相机的视角天然合理，也正好展示"模型在训练视图上看到什么"。
        """
        try:
            import torch

            from ..gsdata.colmap_io import read_model
        except Exception:  # noqa: BLE001
            return None

        job_dir = Path(getattr(ctx, "job_dir", out_dir.parent.parent))
        data_dir = job_dir / "04_train" / "data"
        sparse = data_dir / "sparse"
        model_dir = sparse / "0" if (sparse / "0" / "cameras.bin").exists() else sparse
        if not (model_dir / "cameras.bin").exists():
            ctx.info(f"未找到训练用稀疏模型（{model_dir}），预览退回环绕相机")
            return None
        try:
            rec = read_model(model_dir)
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"读取稀疏模型失败（{type(exc).__name__}），预览退回环绕相机")
            return None
        if not rec.images:
            return None

        poses = []
        for im in rec.images:
            cam = rec.cameras.get(im.camera_id)
            if cam is None:
                continue
            r, t = im.cam_to_world()
            c2w = np.eye(4)
            c2w[:3, :3] = r
            c2w[:3, 3] = t
            poses.append((c2w.astype(np.float32), cam.intrinsics().astype(np.float32),
                          int(cam.width), int(cam.height), im.name))
        # 均匀抽 4 个视角，避免都挤在同一侧
        if len(poses) > 4:
            idx = np.linspace(0, len(poses) - 1, 4).round().astype(int)
            poses = [poses[i] for i in idx]
        ctx.info(f"预览将使用 {len(poses)} 个训练相机位姿（来自 {model_dir.name}）")
        return poses

    def _render_from_poses(self, ctx: StageContext, xyz, scales_real, quats, opac, rgb,
                           poses, out_dir: Path, count: int) -> list[dict]:
        """用给定的真实相机位姿渲染预览（gt 图若存在则并排拼上，便于对照）。

        注意 scales_real 是**真实尺度**（_read_ply 已指数化），不要再 exp。
        """
        import torch
        from gsplat import rasterization
        from PIL import Image

        n = int(xyz.shape[0])
        shs = ((rgb - 0.5) / 0.28209479177387814).reshape(n, 1, 3)
        names = ["front", "side", "top", "iso"]
        out: list[dict] = []
        for k, (c2w, K, cw, ch, img_name) in enumerate(poses[:count]):
            W = H = 512
            sx, sy = W / max(cw, 1), H / max(ch, 1)
            Ks = K.copy()
            Ks[0, 0] *= sx
            Ks[1, 1] *= sy
            Ks[0, 2] *= sx
            Ks[1, 2] *= sy
            viewmat = np.linalg.inv(c2w).astype(np.float32)
            with torch.no_grad():
                colors, alphas, _ = rasterization(
                    means=xyz, quats=quats, scales=scales_real,
                    opacities=opac.clamp(0, 1), colors=shs,
                    viewmats=torch.as_tensor(viewmat, device="cuda")[None],
                    Ks=torch.as_tensor(Ks, device="cuda")[None],
                    width=W, height=H, sh_degree=0, packed=False, render_mode="RGB",
                )
                img = colors[0] + (1 - alphas[0])
            arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

            # 若去畸变图像还在，把真值放在右侧做对照（对诊断很有用）
            gt_p = None
            job_dir = Path(getattr(ctx, "job_dir", out_dir.parent.parent))
            cand = job_dir / "04_train" / "data" / "images" / img_name
            if cand.exists():
                try:
                    with Image.open(cand) as gi:
                        gt = np.asarray(gi.convert("RGB").resize((W, H)))
                    gt_p = np.concatenate([arr, gt], axis=1)
                except Exception:  # noqa: BLE001
                    gt_p = None

            p = out_dir / f"preview_{names[k] if k < len(names) else k}.png"
            Image.fromarray(gt_p if gt_p is not None else arr).save(p)
            kb = round(p.stat().st_size / 1024, 1)
            out.append({"kind": "preview", "path": p.name, "size_kb": kb, "size_mb": round(kb / 1024, 3),
                        "desc": f"训练视角 {img_name} 渲染" + ("（左：渲染，右：真值）" if gt_p is not None else "")})
        if out:
            ctx.info(f"已生成 {len(out)} 张训练视角渲染预览")
        return out

    def _render_previews_scatter(self, ctx: StageContext, gauss: dict, out_dir: Path, count: int = 4) -> list[dict]:
        """兜底：用 matplotlib 做正交投影散点预览（无 torch/CUDA 时使用）。"""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            ctx.warn("未安装 matplotlib，跳过预览图生成")
            return []

        xyz, rgb = gauss["xyz"], gauss["rgb"] / 255.0
        views = [("front", 0, 1), ("side", 0, 2), ("top", 1, 2), ("iso", None, None)]
        out: list[dict] = []
        for name, a, b in views[:count]:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
            if name == "iso":
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                ax.remove()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(xyz[:: max(1, len(xyz) // 60000), 0], xyz[:: max(1, len(xyz) // 60000), 1],
                           xyz[:: max(1, len(xyz) // 60000), 2], c=rgb[:: max(1, len(xyz) // 60000)],
                           s=0.6, marker=".")
                ax.set_axis_off()
            else:
                step = max(1, len(xyz) // 80000)
                ax.scatter(xyz[::step, a], xyz[::step, b], c=rgb[::step], s=0.5, marker=".")
                ax.set_aspect("equal")
                ax.set_axis_off()
            fig.tight_layout(pad=0)
            p = out_dir / f"preview_{name}.png"
            fig.savefig(p, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            # 预览图很小，用 KB 报；字段名与 MB 分开，避免单位混淆
            kb = round(p.stat().st_size / 1024, 1)
            out.append({"kind": "preview", "path": p.name, "size_kb": kb, "size_mb": round(kb / 1024, 3),
                        "desc": f"{name} 视角预览（散点投影）"})
        if out:
            ctx.info(f"已生成 {len(out)} 张预览图（散点投影）")
        return out

    # ------------------------------------------------------------------ 查看器
    def _write_viewer(self, path: Path, gauss: dict, ctx: StageContext) -> Path:
        """生成离线查看器：把点云内嵌进 HTML，用 Canvas 2D 做交互（无外网 CDN 依赖）。"""
        max_pts = 120000
        step = max(1, len(gauss["xyz"]) // max_pts)
        xyz = gauss["xyz"][::step]
        rgb = np.clip(gauss["rgb"][::step], 0, 255).astype(int)
        center = xyz.mean(axis=0)
        xyz = xyz - center
        scale = float(np.abs(xyz).max()) or 1.0
        data = [
            [round(float(x / scale), 4), round(float(-y / scale), 4), round(float(-z / scale), 4),
             int(r), int(g), int(b)]
            for (x, y, z), (r, g, b) in zip(xyz, rgb)
        ]
        payload = json.dumps(data, separators=(",", ":"))
        html = _VIEWER_TEMPLATE.replace("__POINTS__", payload).replace("__COUNT__", str(len(gauss["xyz"])))
        path.write_text(html, encoding="utf-8")
        return path


_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>gsflow 3DGS 查看器</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0d10; color:#e6e8eb; font:14px/1.5 "Segoe UI",system-ui,sans-serif; overflow:hidden; }
  #hud { position:fixed; left:12px; top:12px; background:rgba(20,24,30,.82); border:1px solid #2a3038;
          border-radius:10px; padding:10px 14px; backdrop-filter: blur(6px); }
  #hud b { color:#7dd3fc; }
  #hint { position:fixed; right:12px; bottom:12px; opacity:.65; font-size:12px; }
  canvas { display:block; }
</style>
</head>
<body>
<div id="hud">gsflow 预览 · 高斯总数 <b>__COUNT__</b> · 内嵌采样 <b id="n"></b><br>
  拖拽旋转 · 滚轮缩放 · 右键平移</div>
<div id="hint">本地文件，无需联网</div>
<canvas id="c"></canvas>
<script>
const PTS = __POINTS__;
document.getElementById('n').textContent = PTS.length;
const cv = document.getElementById('c'), g = cv.getContext('2d');
let W=0,H=0, yaw=0.6, pitch=0.35, dist=2.6, panx=0, pany=0, down=false, btn=0, lx=0, ly=0;
function resize(){ const d=devicePixelRatio||1; W=cv.width=innerWidth*d; H=cv.height=innerHeight*d;
  cv.style.width=innerWidth+'px'; cv.style.height=innerHeight+'px'; g.setTransform(d,0,0,d,0,0); }
addEventListener('resize', resize); resize();
function project(p){
  // 先绕 Y 再绕 X
  let [x,y,z]=p;
  let cy=Math.cos(yaw), sy=Math.sin(yaw);
  let x1 = x*cy + z*sy, z1 = -x*sy + z*cy;
  let cp=Math.cos(pitch), sp=Math.sin(pitch);
  let y1 = y*cp - z1*sp, z2 = y*sp + z1*cp;
  const f = 1.9 / (dist + z2 + 0.001);
  return [innerWidth/2 + x1*f*innerWidth*0.45 + panx, innerHeight/2 - y1*f*innerWidth*0.45 + pany, z2];
}
function draw(){
  g.clearRect(0,0,innerWidth,innerHeight);
  g.fillStyle='#0b0d10'; g.fillRect(0,0,innerWidth,innerHeight);
  const order = PTS.map(p => [project(p)[2], p]).sort((a,b)=>b[0]-a[0]);
  for (const [,p] of order){
    const [sx,sy,z]=project(p);
    const r=Math.max(0.7, 2.4/(dist+z+0.9));
    g.fillStyle='rgba('+p[3]+','+p[4]+','+p[5]+',0.92)';
    g.fillRect(sx, sy, r, r);
  }
}
cv.addEventListener('mousedown', e=>{down=true; btn=e.button; lx=e.clientX; ly=e.clientY;});
addEventListener('mouseup', ()=>down=false);
addEventListener('mousemove', e=>{
  if(!down) return;
  const dx=e.clientX-lx, dy=e.clientY-ly; lx=e.clientX; ly=e.clientY;
  if(btn===2){ panx+=dx; pany+=dy; } else { yaw+=dx*0.008; pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.008)); }
  draw();
});
cv.addEventListener('contextmenu', e=>e.preventDefault());
cv.addEventListener('wheel', e=>{ e.preventDefault(); dist=Math.max(0.4, Math.min(12, dist*(1+Math.sign(e.deltaY)*0.08))); draw(); }, {passive:false});
draw();
</script>
</body>
</html>
"""
