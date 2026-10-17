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


def _sh_tensors(gauss: dict, device: str = "cuda"):
    """从 `_read_ply` 的结果取出 (完整 SH 张量, sh_degree)。

    单独抽成函数是为了**只在一处**决定"用 DC 还是用全 SH" ——
    这个决定曾在 4 个渲染点各写一遍，于是全部退化成了 `sh_degree=0`
    （丢高阶 SH ⇒ 实测低估 1.91 dB，见 `_read_ply` 的说明）。
    缺 `shs` 字段时回退为仅 DC（老 PLY / 非 3DGS 来源）。
    """
    import torch

    if gauss.get("shs") is not None:
        shs = torch.as_tensor(gauss["shs"], dtype=torch.float32, device=device)
        return shs, int(gauss.get("sh_degree", 0) or 0)
    rgb = torch.as_tensor(gauss["rgb"], dtype=torch.float32, device=device) / 255.0
    return ((rgb - 0.5) / SH_C0).reshape(-1, 1, 3), 0
GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
COMPONENT_FLOAT = 5126
COMPONENT_UINT = 5125
MODE_POINTS = 0
MODE_TRIANGLES = 4


def _orient_normals_outward(mesh):
    """按"从物体中心向外"统一三角形绕序，使法线一致朝外。

    ## 为什么需要（实测数据）

    `bake_texture_from_views` 用 `facing = -(N·vd) > 0.15` 挑选可用视角。
    法线朝内的面在**任何**视角下都不满足该判据 ⇒ 对应 UV texel 填不上颜色、保持纯黑。

    实测 img3-crop（泊松重建网格，非封闭、`is_orientable()=False`）：
      * 面法线朝外比例仅 **43.6%**；
      * 纹理图集里 **53.96%** 的 UV 覆盖 texel 是黑的 ⇒ 模型出现大片黑斑。
    翻转绕序后朝外比例 **100.0%**，黑洞消失。

    ## 为什么不用 `orient_triangles()`

    对这份非封闭网格它是**空操作**（以前实测翻转率 17.9% → 17.9%）——
    它依赖可定向的流形拓扑，而泊松网格表面自交导致拓扑混乱。

    这里用最直接的判据：面法线与"物体中心→面心"同向即视为朝外。
    对凸/星形物体完全正确；对凹形物体在极少数面上可能判错，
    但**远好于 43.6% 的随机朝向**，且不改变几何形状（只换绕序）。
    """
    import numpy as np

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if len(verts) == 0 or len(tris) == 0:
        return mesh

    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    c = verts.mean(0)
    fc = (p0 + p1 + p2) / 3.0 - c
    fc /= np.maximum(np.linalg.norm(fc, axis=1, keepdims=True), 1e-12)
    need_flip = np.einsum("ij,ij->i", fn, fc) < 0
    if not need_flip.any():
        return mesh

    new_tris = tris.copy()
    new_tris[need_flip] = new_tris[need_flip][:, [0, 2, 1]]     # 交换第 1、2 顶点

    import open3d as o3d

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(verts)
    out.triangles = o3d.utility.Vector3iVector(new_tris)
    if mesh.has_vertex_colors():
        out.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
    out.compute_vertex_normals()
    return out


def _unwrap_uv_with_timeout(unwrap_fn, mesh, tex_size: int, timeout_s: float):
    """带超时地做 UV 展开。超时返回 (None, None, None, 已用秒数)。

    为什么需要这层：`xatlas.parametrize` 是**单个 C 调用，没法从外面中断**。
    实测它在泊松网格上耗时超线性 —— 9,940 面 0.61s，79,808 面 **813s**，
    107,635 面 >20 分钟未完成。如果没有超时保护，导出阶段会**卡死**，
    结果连 manifest.json 都写不出来（已导出的 glb/obj/ply 成了孤儿产物）。

    做法：把展开放进 **daemon 线程**，主线程 `join(timeout)`。
    * 超时就返回 None，让调用方跳过纹理烘焙、继续走完导出；
    * 那个线程是 daemon，进程退出时不会阻止退出；
    * 副作用是它会继续占一个核直到结束 —— 相比"整个导出卡死"这是可接受的取舍，
      而且我们会明确告诉用户"已跳过、可调面数或调大超时"。
    """
    import threading
    import time as _time

    box: dict = {}

    def _work():
        try:
            box["res"] = unwrap_fn(mesh, tex_size=tex_size)
        except Exception as exc:  # noqa: BLE001
            box["err"] = exc

    t0 = _time.time()
    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(None if timeout_s <= 0 else float(timeout_s))
    dt = _time.time() - t0
    if th.is_alive():
        return None, None, None, dt
    if "err" in box:
        raise box["err"]
    m, tris, uv = box.get("res", (None, None, None))
    return m, tris, uv, dt


def validate_glb(path: Path) -> dict:
    """按 glTF 2.0 规范校验我们导出的 GLB，返回 {ok, problems, info}。

    为什么要单独做：GLB 是二进制容器，"文件有内容"并不代表能被查看器读出来。
    这里检查 magic/版本/总长、chunk 结构、bufferView 的 4 字节对齐与越界、
    POSITION 的 min/max，并按图元模式做针对性校验：

      * `POINTS`（点云式 glb）：POSITION 与 COLOR_0 顶点数必须一致
      * `TRIANGLES`（网格 glb）：必须带 indices，且索引不能越界、
        索引数必须是 3 的倍数（否则有残缺三角形）
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
    mode = prim.get("mode", MODE_TRIANGLES)
    if mode not in (MODE_POINTS, MODE_TRIANGLES):
        problems.append(f"图元模式既不是 POINTS 也不是 TRIANGLES（{mode}）")
    if pos["count"] != col["count"]:
        problems.append("POSITION 与 COLOR_0 顶点数不一致")

    tri_count = 0
    if mode == MODE_TRIANGLES:
        idx_id = prim.get("indices")
        if idx_id is None:
            problems.append("TRIANGLES 图元缺少 indices")
        else:
            idx = gltf["accessors"][idx_id]
            if idx["count"] % 3 != 0:
                problems.append(f"索引数 {idx['count']} 不是 3 的倍数")
            if idx["componentType"] not in (5123, 5125):
                problems.append(f"索引 componentType 应为 UNSIGNED_SHORT/INT（{idx['componentType']}）")
            tri_count = idx["count"] // 3
            # 索引上界必须落在顶点数范围内（越界会让查看器崩溃/黑屏）
            bv = gltf["bufferViews"][idx["bufferView"]]
            seg = raw[chunks[b_idx][2] + bv["byteOffset"]:
                      chunks[b_idx][2] + bv["byteOffset"] + bv["byteLength"]]
            fmt = "<H" if idx["componentType"] == 5123 else "<I"
            step = 2 if idx["componentType"] == 5123 else 4
            n = min(idx["count"], len(seg) // step)
            vals = struct.unpack(f"<{n}{fmt[1]}", seg[:n * step]) if n else ()
            if vals and max(vals) >= pos["count"]:
                problems.append(f"索引越界：max={max(vals)} >= 顶点数 {pos['count']}")

    info = {
        "bytes": len(raw),
        "chunks": len(chunks),
        "mode": mode,
        "count": pos["count"],
        "triangles": tri_count,
        "materials": len(gltf.get("materials", [])),
        "extensions": gltf.get("extensionsUsed", []),
    }
    return {"ok": not problems, "problems": problems, "info": info}


def _gaussian_normals(gauss: dict, ctx) -> object:
    """从高斯的旋转四元数取出"局部 +z 轴在世界系的方向"，当作法线初值。

    为什么比纯几何估计好：3DGS 训练时会把高斯压扁并**对齐表面**
    （这正是它能重建几何的原因），所以旋转分量里就带着表面朝向信息。
    在薄壳 / 纹理弱的物体上，这比 `estimate_normals` 的 PCA 估计稳得多，
    而泊松重建的质量**几乎完全取决于法线朝向是否正确**。

    返回 None 表示拿不到（调用方会退回几何估计）。
    """
    try:
        import numpy as np

        q = np.asarray(gauss["quats"], dtype=np.float64)
        pts = np.asarray(gauss["xyz"])
        if q.ndim != 2 or q.shape[1] != 4 or len(q) != len(pts):
            return None
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # 四元数 (w,x,y,z) 对应旋转矩阵的第三列
        n = np.stack([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y),
        ], axis=1)
        nrm = np.linalg.norm(n, axis=1, keepdims=True)
        if not np.all(np.isfinite(nrm)) or float(nrm.min()) < 1e-9:
            return None
        ctx.info("法线初值取自高斯旋转（局部 +z 轴）")
        return n / nrm
    except Exception:  # noqa: BLE001
        return None


class ExportStage(Stage):
    name = "export"
    title = "⑤ 导出与预览"
    description = "把 point_cloud.ply 转成 glb / spz，并生成离线查看器与预览图"
    requires = ("train",)
    optional = True
    inputs = (Artifact("04_train/point_cloud", "训练产物目录"),)
    outputs = (Artifact("05_export/manifest.json", "导出清单"),)

    def run(self, ctx: StageContext) -> StageResult:
        # `export.ply` 可显式指定输入点云 —— 这条曾**只写在报错提示里、代码却没读**，
        # 导致 `--set export.ply=...` 静默无效（本次排查纹理时踩到）。
        # 用途：用"已过滤掉内部面的点云/网格"再导出，而不重跑训练。
        ply = None
        explicit = str(ctx.cfg("ply", "") or "").strip()
        if explicit:
            cand = Path(explicit)
            if not cand.is_absolute():
                cand = (ctx.job_dir / cand) if (ctx.job_dir / cand).exists() else (Path.cwd() / cand)
            if cand.exists():
                ply = cand
                ctx.info(f"使用显式指定的输入：{explicit}")
            else:
                ctx.warn(f"export.ply 指定的文件不存在：{explicit}（回退自动查找）")
        if ply is None:
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

        # ⚠️ 在"重活"（网格提取 + 纹理烘焙）之前先落一次清单。
        # 踩坑：manifest 原来只在**全部步骤之后**写。而纹理烘焙在大网格上会卡很久
        # （xatlas UV 展开实测 79808 面要 813 秒），一旦超时被中断，
        # 已经导出的 glb/obj/ply 就成了**没有清单的孤儿产物** ——
        # GUI 的产物列表、脚本的产物发现全靠 manifest，等于白干。
        # 所以：每完成一批产物就刷新清单，"部分成功"也要留下可用记录。
        def _flush_manifest(extra_note: str = "") -> None:
            m = {
                "source_ply": str(ply),
                "gaussians": int(len(gauss["xyz"])),
                "formats": formats,
                "artifacts": produced,
                "complete": False,
            }
            if extra_note:
                m["note"] = extra_note
            (out_dir / "manifest.json").write_text(
                json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

        _flush_manifest("点云类产物已就绪；网格/贴图进行中")

        # 三角网格导出（用户反馈"点云没有实感"）
        # 深度反投影 -> 泊松 -> 平滑 -> QEM 简化；详见 gsdata/mesh_export.py 顶部注释。
        if "mesh" in formats or "obj_mesh" in formats or "mesh_obj" in formats:
            try:
                mesh_items = self._export_mesh(ctx, ply, out_dir)
                produced.extend(mesh_items)
                _flush_manifest("网格已就绪")
            except Exception as exc:  # noqa: BLE001
                # 记录**完整 traceback**：只记 `type: msg` 时，
                # "array is not broadcastable to correct shape" 这类错误完全无法定位
                # （本轮因此多绕了一轮）。把最后几帧写进日志，直接看行号。
                import traceback as _tb

                ctx.warn(f"网格导出失败（不影响其它产物）：{type(exc).__name__}: {exc}")
                for _line in _tb.format_exc().strip().splitlines()[-6:]:
                    ctx.warn(f"  {_line}")
                _flush_manifest(f"网格导出失败：{type(exc).__name__}")

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
            "complete": True,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return StageResult(artifacts=[str(out_dir / "manifest.json")], metrics=manifest,
                           message=f"导出 {len(produced)} 个产物")

    # ------------------------------------------------------------------ 读取
    @staticmethod
    def _find_ply(job_dir: Path) -> Path | None:
        """挑要导出的 point_cloud.ply —— **优先挑指标最佳的检查点，而不是最后一个**。

        为什么不是"最后一个"：3DGS 在小规模环绕拍摄上常见"过训练"——
        密度控制后期不断加密，拟合反而变差。真实素材实测：
            step 8000  PSNR 26.45 dB  ← 峰值
            step 15000 PSNR 21.26 dB  ← 退化 5.2 dB
        而早期实现固定导出 metrics.json 里的 `ply` 字段（= 最后一次迭代），
        等于**每次都导出最差的那个模型**。

        现在的选择顺序：
          1. metrics.json 里记录的 `best_ply`（训练时按评估 PSNR 选出的最佳检查点）；
          2. 若有 psnr_curve，按曲线峰值对应的迭代去取同目录的检查点；
          3. 否则回退到迭代次数最高的（并保留按数值而非字典序排序的修正）。

        注意：不能按路径字符串排序——`iteration_5000` 会排在 `iteration_30000` 之后。
        """
        import re

        base = job_dir / "04_train" / "point_cloud"
        metrics = job_dir / "04_train" / "metrics.json"
        data: dict = {}
        if metrics.exists():
            try:
                data = json.loads(metrics.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, TypeError):
                data = {}

        def iteration(p: Path) -> int:
            m = re.search(r"iteration_(\d+)", str(p))
            return int(m.group(1)) if m else -1

        # 1) 训练阶段直接记下的最佳检查点
        #    注意 Path("") 会解析成当前目录且 .exists() 为真，必须先用字符串长度兜住，
        #    否则缺字段时会返回一个空路径。
        best_raw = str(data.get("best_ply") or "").strip()
        if best_raw:
            best = Path(best_raw)
            if best.exists():
                return best

        # 2) 用 PSNR 曲线的峰值迭代去取
        curve = data.get("psnr_curve") or []
        if curve and base.exists():
            try:
                peak_step = max(curve, key=lambda x: float(x.get("psnr", 0))).get("step")
                cand = base / f"iteration_{int(peak_step)}" / "point_cloud.ply"
                if cand.exists():
                    return cand
                # 峰值迭代没存检查点：退而求其次，取不超过峰值迭代里最大的那个
                allp = sorted(base.rglob("point_cloud.ply"), key=iteration)
                below = [p for p in allp if iteration(p) <= int(peak_step)]
                if below:
                    return below[-1]
            except (TypeError, ValueError):
                pass

        # 3) 回退：迭代次数最高
        if not base.exists():
            return None
        cands = list(base.rglob("point_cloud.ply"))
        if not cands:
            return None
        return max(cands, key=iteration)

    @staticmethod
    def _read_ply(ply: Path, max_points: int = 0) -> dict:
        """读取 3DGS PLY（**委托给 `gsflow.gsdata.shio.read_gaussians_full`**）。

        **重要修正（用户指出的"实现被简化"）**：
        旧版只解 `f_dc_*` 成 RGB 并**丢掉全部高阶 SH**。但训练用 `sh_degree=3`
        ⇒ PLY 里有 45 个 `f_rest_*`。丢掉它们意味着：
          * 预览图 / 网格顶点色**不是模型的真实外观**（缺视角相关的色彩变化）；
          * 用 `sh_degree=0` 渲染来评估，实测**低估 1.91 dB**
            （同模型：全 SH 22.10 dB vs 仅 DC 20.20 dB），
            而且对"改善视角相关细节"的优化手段（感知损失等）**天然不敏感**
            —— 这会直接制造**假阴性**（把有效的优化判成无效）。

        返回值：
          * `rgb`        —— DC 解出的视角无关基色（GLB/OBJ 顶点色用）
          * `shs`        —— **完整** SH [N, (deg+1)², 3]，供 gsplat 真实渲染
          * `sh_degree`  —— 由 `f_rest_*` 数量反推（45 ⇒ 3）

        实现只在 `gsdata/shio.py` 一处 —— 本会话已因"同一决定写 4 遍"而丢过高阶 SH，
        所以这里不再自己解析，直接委托，避免再次分叉。
        """
        from ..gsdata.shio import read_gaussians_full

        return read_gaussians_full(ply, max_points=max_points)
    # ------------------------------------------------------------------ 网格
    def _export_mesh(self, ctx: StageContext, ply: Path, out_dir: Path) -> list[dict]:
        """用 TSDF 融合从训练好的 3DGS 提取三角网格（`.obj` + 可选 `.ply`）。

        流程：读取 PLY 的高斯 -> 取训练时的真实相机位姿 -> 渲染 RGB+期望深度
              -> open3d ScalableTSDFVolume 融合 -> 去离群 -> 写 obj。
        参数（都有合理默认，可用 --set 覆盖）：
              export.mesh_voxel_div   体素细分（场景尺度/该值），默认 512，越小越粗越快
              export.mesh_max_views   最多融合多少个视角，默认 80
              export.mesh_width       渲染分辨率（正方形边长），默认 640
              export.mesh_min_alpha   低于该不透明度的像素不参与融合，默认 0.35
              export.mesh_largest_only 只保留最大连通块，默认 true（单物体）
        """
        import numpy as np
        import torch

        from ..gsdata.mesh_export import (_scene_extent, auto_crop_bounds, clean_mesh,
                                          filter_by_multiview_consistency,
                                          gaussian_axis_normals, poisson_mesh,
                                          to_largest_component,
                                          transfer_normals_from_gaussians)

        gauss = self._read_ply(ply, max_points=0)
        n = len(gauss["xyz"])
        method = str(ctx.cfg("mesh_method", "poisson") or "poisson").lower()
        ctx.info(f"网格导出：方法={method}，读取 {n} 个高斯")

        # ---- 高斯自带法线（实验开关，**默认关闭**）----
        # 动机：高斯法线的**局部**一致度（0.787 / 翻转 0.2%）远好于深度梯度法线（翻转 ~17%）。
        # ⚠️ 但实测**用它会明显变差**，所以默认不开（`mesh_normals=depth`）：
        #   poisson 路线基准（深度梯度法线）翻转 43.4%；
        #   迁移 24% 的高斯法线后 -> 45.4%；迁移 39% -> 45.6%（还试过朝向对齐，无效）。
        # 原因：高斯的 +z 轴是**该高斯的朝向**，而泊松需要的是**一致定向的切线平面法线**。
        # 局部一致 ≠ 全局可积；混入 24~39% 不同来源的法线会让泊松的全局解更差。
        # 因此保留开关供后续研究（例如先做全局定向再迁移），但不作为默认。
        gauss_nrm = None
        if str(ctx.cfg("mesh_normals", "depth") or "depth").lower() == "gaussian":
            try:
                gauss_nrm = gaussian_axis_normals(np.asarray(gauss["quats"], dtype=np.float64))
            except Exception as exc:  # noqa: BLE001
                ctx.warn(f"高斯法线不可用（{type(exc).__name__}: {exc}），退回深度梯度法线")
                gauss_nrm = None

        # 先把网格提取聚焦到"主体"上：COLMAP 会把桌面/背景杂物也重建出来。
        # 泊松对远处杂点比 TSDF 更敏感（会生成巨大的"鼓包"），所以默认裁得更紧。
        xyz_np = np.asarray(gauss["xyz"], dtype=np.float64)
        op_np = np.asarray(gauss["opacity"], dtype=np.float64).reshape(-1)
        # 自动聚焦默认分位 0.95：实测 0.90 偏紧（会切掉主体部分外缘），
        # 同一模型同一验证脚本下轮廓 IoU 0.95 明显更好（0.750 vs 0.696）。
        default_q = 0.95 if method == "poisson" else 0.98
        if ctx.cfg("mesh_auto_crop", True):
            lo, hi, cinfo = auto_crop_bounds(
                xyz_np, op_np,
                op_thresh=float(ctx.cfg("mesh_crop_opacity", 0.3) or 0.3),
                quantile=float(ctx.cfg("mesh_crop_quantile", default_q) or default_q),
            )
            inside = np.all((xyz_np >= lo) & (xyz_np <= hi), axis=1)
            if int(inside.sum()) >= 256:
                ctx.info(f"网格聚焦：保留 {int(inside.sum())}/{n} 个高斯"
                         f"（裁剪框尺寸 {np.round(hi - lo, 3).tolist()}）")
                gauss = {k: (v[inside] if hasattr(v, "__len__") and len(v) == n else v)
                         for k, v in gauss.items()}
                xyz_np, op_np = xyz_np[inside], op_np[inside]
                n = len(gauss["xyz"])
            else:
                ctx.warn("聚焦后剩余高斯过少，改用全部点")

        # ---------------- 路线一：深度反投影 -> 泊松 ----------------
        # 用户给的社区标准流程：
        #   1) 从 3DGS 渲染深度图，按不透明度/深度一致性过滤后**反投影**成稠密点云
        #   2) 统计去噪 + **体素下采样**均匀化密度 + 法线一致定向
        #   3) 泊松重建（depth 控制八叉树层数）
        #   4) 后处理：裁掉低密度包络面 + 平滑 + QEM 简化
        # 关键点：**不能直接拿高斯中心当输入点云** —— 那些点是一团一团分布在
        # 表面附近的三维体，泊松会得到厚壳与鼓包；反投影出来的才是"薄薄一层"。
        #
        # 两种模式共用这条主链，差别只在"反投影之后是否做多视角一致性筛选"：
        #   * `poisson`（默认）：沿用社区标准流程，适合视图较少/深度较干净的场景。
        #   * `2dgs`：额外做多视角深度一致性投票（见下面 consistency 段），
        #     针对本项目实测的"表面毛刺 + 法线翻转 35%"问题。
        if method in ("poisson", "poisson_depth", "backproject", "2dgs"):
            from ..gsdata.mesh_export import (backproject_depth_to_points,
                                              render_depths, select_views_by_distance,
                                              simplify_mesh, smooth_mesh, voxel_downsample)

            cams = self._training_cameras(
                ctx, out_dir, limit=int(ctx.cfg("mesh_max_views", 80) or 80))
            if not cams:
                ctx.warn("找不到训练相机位姿，无法反投影深度（需要 04_train/data/sparse）")
                return []

            c2ws = np.stack([c[0] for c in cams])
            src_wh = (int(cams[0][2]), int(cams[0][3]))
            ks = np.stack([c[1] for c in cams])
            pts_all = np.asarray(gauss["xyz"], dtype=np.float64)

            if ctx.cfg("mesh_view_filter", True):
                keep, vstat = select_views_by_distance(c2ws, pts_all)
                if int(keep.sum()) < len(keep):
                    ctx.info(f"视角筛选：保留 {vstat['kept']}/{vstat['total']}"
                             f"（距离中位数 {vstat['median_dist']}，MAD {vstat['mad']}）")
                    c2ws, ks = c2ws[keep], ks[keep]
                    cams = [c for c, k in zip(cams, keep) if k]

            center, extent = _scene_extent(c2ws, pts_all)
            width = int(ctx.cfg("mesh_width", 640) or 640)

            xyz = torch.as_tensor(gauss["xyz"], dtype=torch.float32, device="cuda")
            scales_real = torch.as_tensor(gauss["scales"], dtype=torch.float32, device="cuda")
            quats_t = torch.as_tensor(gauss["quats"], dtype=torch.float32, device="cuda")
            opac = torch.as_tensor(gauss["opacity"], dtype=torch.float32, device="cuda").reshape(-1)
            rgb = torch.as_tensor(gauss["rgb"], dtype=torch.float32, device="cuda") / 255.0
            # 用**完整 SH**（含高阶）渲染：只用 DC 会低估约 1.9 dB（见 _read_ply 注释），
            # 而深度反投影/预览都应以模型真实外观为准。
            shs, sh_deg = _sh_tensors(gauss, device="cuda")
            ctx.info(f"渲染用 SH 阶数 = {sh_deg}"
                     + ("（含高阶，与训练一致）" if sh_deg > 0 else "（仅 DC）"))

            ctx.info(f"① 渲染深度：{min(len(c2ws), int(ctx.cfg('mesh_max_views', 80) or 80))}"
                     f"/{len(c2ws)} 个视角 @ {width}px ...")
            rendered, used_idx = render_depths(
                means=xyz, quats=quats_t, scales_real=scales_real,
                opacities=opac.clamp(0, 1), shs=shs,
                c2ws=c2ws, ks=ks, width=width, height=width, sh_degree=sh_deg,
                src_wh=src_wh, max_views=int(ctx.cfg("mesh_max_views", 80) or 80),
            )

            ctx.info("② 深度反投影 -> 稠密点云 ...")
            bp_pts, bp_cols, bp_nrm, bp_stat = backproject_depth_to_points(
                rendered=rendered, c2ws=c2ws, view_idx=used_idx, ks=ks, src_wh=src_wh,
                width=width, height=width,
                # 用 cfgt 而不是 `... or 默认值`：`or` 会把合法的 0 当成缺省，
                # 于是"显式配成 0 = 关闭该步"永远无法生效（mesh_depth_stride 就踩过）。
                depth_stride=ctx.cfgt("mesh_depth_stride", 2, int),
                min_alpha=ctx.cfgt("mesh_min_alpha", 0.5, float),
                erode_px=ctx.cfgt("mesh_erode_px", 1, int),
                grad_thresh_ratio=ctx.cfgt("mesh_grad_thresh", 0.06, float),
            )
            if len(bp_pts) < 1000:
                # ⚠️ 这里必须**报错**而不是 warn。踩坑：背面剔除方向写反时，
                # 每个视角的点都被剔掉、反投影恒为 0 点，而这里只 warn 就继续，
                # 结果是"作业显示成功、却没有网格产物"——静默失败极难定位
                # （我就是在核对产物时才发现的）。
                bstat = bp_stat or {}
                raise RuntimeError(
                    f"深度反投影只得到 {len(bp_pts)} 个点，无法重建网格"
                    f"（视角 {bstat.get('views_used', '?')}，有效像素 {bstat.get('pixels', '?')}）。"
                    "常见原因：① 该检查点渲染出的 alpha 普遍低于 mesh_min_alpha；"
                    "② 深度图为空/全零；③ 法线朝向与相机约定不符导致背面剔除删掉全部点"
                    "（可用 scripts\\diag_backproject_exact.py 定位）")
            ctx.info(f"   反投影得到 {len(bp_pts)} 个点"
                     f"（{bp_stat['views_used']} 视角，{bp_stat['pixels']} 个有效像素）")

            # 体素下采样尺寸：均匀化密度（近处视角点密、远处稀疏，差异可达百倍）。
            # 反投影会产出百万级点，必须下采样，否则泊松既慢又过拟合。
            # 默认取 体素尺度 = 场景尺度/400 的 1/4（比几何体素更粗），
            # 实测 250 万点 -> 155 万点，网格质量不受影响。
            # 注意：**在一致性投票之前算**，因为投票体素要按它来定标。
            vds = float(ctx.cfg("mesh_voxel_downsample", 0.0) or 0.0)
            if vds <= 0:
                vds = extent / float(ctx.cfg("mesh_voxel_div", 400.0) or 400.0) * 0.25

            # ---- 「2dgs」模式的核心：多视角深度一致性投票 ----
            #
            # `mesh_method=2dgs` 与默认 `poisson` 走**同一条主链**，差别只在这里：
            # 把"只被单个视角看到的点"清掉。为什么必须清（都是本项目的实测问题）：
            #   * 表面毛刺/薄片：深度不连续处（前景边缘拖尾）产生孤立点，
            #     泊松在局部把它们连成尖刺。
            #   * **法线翻转约 35%**：同一表面被相邻视角各投一份点，两份法线因噪声
            #     方向相反，体素下采样把它们**平均成接近零**，于是光照大片发黑。
            # 判据：体素投票，要求 >= min_views 个视角支持；
            # 并可再按"与该体素平均法线的夹角"剔除方向矛盾的点。
            #
            # 这对应 2DGS / Gaussian Opacity Fields 用"多视角一致性约束表面"的思想，
            # 但作用在**导出阶段**，不需要重训模型。
            if method == "2dgs":
                bp_pts, bp_cols, bp_nrm, vstat = backproject_depth_to_points(
                    rendered=rendered, c2ws=c2ws, view_idx=used_idx, ks=ks, src_wh=src_wh,
                    width=width, height=width,
                    depth_stride=ctx.cfgt("mesh_depth_stride", 2, int),
                    min_alpha=ctx.cfgt("mesh_min_alpha", 0.5, float),
                    erode_px=ctx.cfgt("mesh_erode_px", 1, int),
                    grad_thresh_ratio=ctx.cfgt("mesh_grad_thresh", 0.06, float),
                    return_view_id=True,
                )
                # 投票体素要比下采样体素**大一些**，否则相邻视角对同一表面的采样
                # 永远落不进同一格，投票就没意义。默认 1.5 倍。
                cvoxel = vds * ctx.cfgt("mesh_consistency_voxel_scale", 3.0, float)
                bp_pts, bp_nrm, bp_cols, cstat = filter_by_multiview_consistency(
                    bp_pts, normals=bp_nrm, colors=bp_cols,
                    view_id=vstat.get("view_id"),
                    voxel=cvoxel,
                    min_views=ctx.cfgt("mesh_consistency_min_views", 2, int),
                    normal_dot_min=ctx.cfgt("mesh_consistency_normal_dot", 0.0, float),
                )
                # ⚠️ 这里不能用 `info` —— 它要等下面的 `poisson_mesh(...)` 才被赋值，
                # 提前引用会 UnboundLocalError（实测踩过）。先用局部变量存着，后面合并。
                consistency_info = {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in cstat.items()}
                ctx.info(
                    f"   多视角一致性投票（体素 {cvoxel:.5f}，要求 >= "
                    f"{cstat['min_views']}/{cstat['views_total']} 视角）："
                    f"{cstat['points_in']} -> {cstat['points_out']} 点"
                    f"（保留 {cstat['kept_ratio']*100:.1f}%；"
                    f"视角不足剔除 {cstat['dropped_few_views']}，"
                    f"法线矛盾剔除 {cstat['dropped_normal_conflict']}）")
                if len(bp_pts) < 1000:
                    ctx.warn("一致性投票后点太少，退回不做投票的结果")
                    bp_pts, bp_cols, bp_nrm, _ = backproject_depth_to_points(
                        rendered=rendered, c2ws=c2ws, view_idx=used_idx, ks=ks,
                        src_wh=src_wh, width=width, height=height,
                        depth_stride=ctx.cfgt("mesh_depth_stride", 2, int),
                        min_alpha=ctx.cfgt("mesh_min_alpha", 0.5, float),
                        erode_px=ctx.cfgt("mesh_erode_px", 1, int),
                        grad_thresh_ratio=ctx.cfgt("mesh_grad_thresh", 0.06, float),
                    )
                    consistency_info = None

            bp_pts, bp_cols, bp_nrm = voxel_downsample(bp_pts, bp_cols, bp_nrm, voxel=vds)
            ctx.info(f"③ 体素下采样（{vds:.5f}）后剩 {len(bp_pts)} 个点")

            # ---- 用高斯法线替换深度梯度法线（关键质量改动）----
            # 深度梯度法线的翻转率在 poisson 路线上实测 43%，是"网格碎/毛刺"的直接来源；
            # 高斯自带法线只有 0.2%。把每个反投影点关联到最近的高斯并改用它的法线。
            if gauss_nrm is not None:
                _gx = np.asarray(gauss["xyz"], dtype=np.float64)
                # ⚠️ 必须**现算**：`gauss` 会被上面的"网格聚焦"裁剪（实测 166892 -> 163195），
                # 而在读取时预先算好的法线仍是旧长度，于是下游出现
                # `ValueError: array is not broadcastable to correct shape`。
                # （第一次实现就是在读取处算好再传下来，踩了这个坑。）
                _gn = gaussian_axis_normals(np.asarray(gauss["quats"], dtype=np.float64))
                # 搜索半径：高斯是**体**、反投影点是**面采样**，点到最近高斯中心的距离
                # 通常大于一个体素。实测 `voxel*1.75` 只命中 **3.9%** 的点（太严），
                # 改为可按体素倍数配置，默认 4 倍。
                _nvox = float(ctx.cfgt("mesh_normal_transfer_voxels", 4.0, float))
                tr_nrm, tr_stat = transfer_normals_from_gaussians(
                    bp_pts, _gx, _gn,
                    voxel=vds,
                    max_dist=vds * max(1.0, _nvox),
                    fallback=bp_nrm,
                    # 朝向必须对齐到既有法线场（深度梯度法线已统一为"朝向相机"）。
                    # 不做这一步，混入"朝外"的高斯法线会让翻转率从 20% 恶化到 45%。
                    orient_ref=bp_nrm,
                )
                ctx.info(f"③b 法线改用高斯 +z 轴：{tr_stat['transferred']}/{len(bp_pts)} 点"
                         f"（{tr_stat['transfer_ratio'] * 100:.1f}%），"
                         f"其余 {tr_stat['dropped']} 点退回深度梯度法线；"
                         f"最近高斯距离 p50={tr_stat['dist_voxels_p50']:.2f} "
                         f"p90={tr_stat['dist_voxels_p90']:.2f} "
                         f"p99={tr_stat['dist_voxels_p99']:.2f} 体素，"
                         f"半径上限 {tr_stat['max_dist'] / tr_stat['voxel']:.2f} 体素")
                bp_nrm = tr_nrm

            mesh, info = poisson_mesh(
                bp_pts, bp_cols, normals=bp_nrm,
                depth=int(ctx.cfg("mesh_poisson_depth", 9) or 9),
                # 默认 0.001 —— 不是 0（等于关掉密度裁剪），也不是 0.03。
                # 实测（同一模型、5 视角、224px 的 tune_mesh 扫描）：
                #   0.000 -> IoU 0.725 | 0.001 -> 0.750 | 0.005 -> 0.614 | 0.030 -> 0.315
                # 0.03 这种"看起来合理"的分位数会把大量**真实表面**一起裁掉；
                # 泊松在"完全无观测"处的密度接近 0，所以阈值必须取得很小。
                density_quantile=float(ctx.cfg("mesh_density_quantile", 0.001) or 0.0),
                # 按"到输入点的距离"裁掉泊松桥接出来的面（自适应阈值 = 输入点间距 × sigma）
                trim_distance_sigma=float(ctx.cfg("mesh_trim_sigma", 2.5) or 0.0),
                # `orient=False`：反投影现在给出的法线是**由深度梯度算出的真实表面法线、
                # 且已按"朝向相机"统一过**，不需要（也不应该）再跑一次
                # `orient_normals_consistent_tangent_plane` —— 那一步是给**无朝向**的
                # 原始点云用的，在带噪薄壳上会打乱本来正确的法线，而且很慢。
                orient=False,
            )
            # 2dgs 模式的一致性统计（在 poisson_mesh 之后才有 info 可写）
            if method == "2dgs" and locals().get("consistency_info"):
                info["consistency"] = consistency_info
            info["method"] = "poisson_backproject" if method != "2dgs" else "2dgs_consistency"
            info["points_used"] = int(len(bp_pts))
            ctx.info(f"④ 泊松重建（depth={info['depth']}）："
                     f"{len(mesh.triangles)} 个三角形 / {len(mesh.vertices)} 个顶点")

            # ---- ⑤⑥⑦ 后处理链：**预简化 -> 平滑 -> 最终 QEM** ----
            # 顺序的语义与原因写在 `postprocess_mesh` 的 docstring 里（不要调换）：
            # 平滑必须跑在"已被预简化界定规模"的网格上，否则泊松 depth 一高就会卡死
            # （实测 2771 万面时旧顺序跑了 61 分钟仍未完成）。
            from ..gsdata.mesh_export import postprocess_mesh

            target = int(ctx.cfg("mesh_target_triangles", 400000) or 0)
            mesh, _steps = postprocess_mesh(
                mesh,
                target_triangles=target,
                smooth_iters=int(ctx.cfg("mesh_smooth_iters", 5) or 0),
                smooth_method=str(ctx.cfg("mesh_smooth", "taubin") or "taubin"),
                presimplify_ratio=float(ctx.cfg("mesh_presimplify_ratio", 10.0) or 0.0),
                log=ctx.info,
            )
            # ---- 删除"从未被任何相机看到"的内部面（泊松重建的垃圾几何）----
            #
            # 实测 img3-crop：**25.6% 的面从未被任何视角看到**，且它们的
            # 包围盒 ≈ 全网格、到中心距离 ≈ 全网格 ⇒ **散布在物体内部**，
            # 不是背面遮挡。害处：
            #   * 在 UV 展开里白占 texture 空间却烘不出颜色 ⇒ 纹理黑洞
            #     （实测 43.7% 的 chart 内部 texel 是黑的）；
            #   * 让网格法线混乱、体积虚胖；对渲染零贡献。
            #
            # ⚠️ 判据用"是否被相机看到过"，**不用**"落在前景掩码内的比例"——
            # 后者在转台素材上原理错误（物体旋转使同一面在掩码内外来回变），
            # 我曾据此误删 28.7% 顶点，回原始帧才发现删的是物体细节（见 docs/03 G96）。
            if bool(ctx.cfg("mesh_drop_unseen", True)):
                try:
                    from ..gsdata.mesh_export import drop_unseen_faces

                    imgs_b, c2ws_b, ks_b, whs_b = self._load_bake_inputs(ctx, cams)
                    mesh, dinfo = drop_unseen_faces(
                        mesh, c2ws_b, ks_b, whs_b,
                        min_views=int(ctx.cfg("mesh_min_views", 1) or 1),
                        log=ctx.info)
                    info = {**info, "drop_unseen": dinfo}
                except Exception as exc:  # noqa: BLE001
                    ctx.warn(f"删除不可见面失败（{type(exc).__name__}: {exc}），继续导出")
            # ---- 最后把法线统一朝外（**必须在简化/删除面之后**）----
            #
            # 踩过的坑：我最初把定向放在烘焙前（`_finish_mesh` 内部），
            # 但那是在 QEM 简化**之后**又删过面 —— 两者都会打乱面绕序，
            # 于是导出网格的"朝外比例"只有 **85.1%**（验收时为 FAIL）。
            # 实测：在同一份网格上补一次定向能到 100%（翻转 7,270 个面）。
            # 放在这里可以把"导出网格"与"烘焙输入"一起受益。
            try:
                mesh = _orient_normals_outward(mesh)
                ctx.info("网格后处理：法线已统一朝外")
            except Exception as exc:  # noqa: BLE001
                ctx.warn(f"法线定向失败（{type(exc).__name__}: {exc}）")
            return self._finish_mesh(ctx, mesh, info, out_dir, cams=cams)

        # ---------------- 路线二：TSDF 融合 ----------------
        from ..gsdata.mesh_export import fuse_tsdf

        # 网格融合需要尽量多的视角覆盖表面（预览才只需要 4 个）
        cams = self._training_cameras(ctx, out_dir, limit=int(ctx.cfg("mesh_max_views", 80) or 80))
        if not cams:
            ctx.warn("找不到训练相机位姿，无法做 TSDF 融合（需要 04_train/data/sparse）")
            return []

        c2ws = np.stack([c[0] for c in cams])
        # 内参对应的原始尺寸（_training_cameras 已带回 cam.width/height）
        src_wh = (int(cams[0][2]), int(cams[0][3]))
        ks = np.stack([c[1] for c in cams])
        pts = np.asarray(gauss["xyz"], dtype=np.float64)

        # 剔除距离异常的视角（贴太近 / 拉太远），否则场景尺度被放大、TSDF 全部失配
        if ctx.cfg("mesh_view_filter", True):
            keep, vstat = select_views_by_distance(c2ws, pts)
            if int(keep.sum()) < len(keep):
                ctx.info(f"视角筛选：保留 {vstat['kept']}/{vstat['total']}"
                         f"（距离中位数 {vstat['median_dist']}，MAD {vstat['mad']}）")
                c2ws, ks = c2ws[keep], ks[keep]
                cams = [c for c, k in zip(cams, keep) if k]

        center, extent = _scene_extent(c2ws, pts)
        ctx.info(f"场景中心 {np.round(center, 3).tolist()}；尺度 {extent:.3f}")

        width = int(ctx.cfg("mesh_width", 640) or 640)
        height = width
        max_views = int(ctx.cfg("mesh_max_views", 80) or 80)
        # 体素尺寸按场景尺度自适应：extent / mesh_voxel_div
        # （社区实现常用绝对值 0.002，但那是针对具体数据集的；这里自适应更通用）
        voxel_div = float(ctx.cfg("mesh_voxel_div", 400.0) or 400.0)
        min_alpha = float(ctx.cfg("mesh_min_alpha", 0.35) or 0.35)
        largest_only = ctx.cfg("mesh_largest_only", True)
        if isinstance(largest_only, str):
            largest_only = largest_only.strip().lower() in ("1", "true", "yes", "on")

        xyz = torch.as_tensor(gauss["xyz"], dtype=torch.float32, device="cuda")
        scales_real = torch.as_tensor(gauss["scales"], dtype=torch.float32, device="cuda")
        quats = torch.as_tensor(gauss["quats"], dtype=torch.float32, device="cuda")
        opac = torch.as_tensor(gauss["opacity"], dtype=torch.float32, device="cuda").reshape(-1)
        shs, sh_deg = _sh_tensors(gauss, device="cuda")

        ctx.info(f"渲染深度：{min(max_views, len(c2ws))}/{len(c2ws)} 个视角 @ {width}px "
                 f"（SH 阶数 {sh_deg}）...")
        rendered, used_idx = render_depths(
            means=xyz, quats=quats, scales_real=scales_real,
            opacities=opac.clamp(0, 1), shs=shs,
            c2ws=c2ws, ks=ks,
            width=width, height=height, sh_degree=sh_deg,
            src_wh=src_wh, max_views=max_views,
        )

        ctx.info(f"TSDF 融合（体素 = 尺度/{voxel_div:.0f} = {extent/voxel_div:.5f}）...")
        mesh, info = fuse_tsdf(
            rendered=rendered, c2ws=c2ws, view_idx=used_idx,
            ks=ks, width=width, height=height, src_wh=src_wh,
            center=center, extent=extent,
            voxel_size=extent / max(voxel_div, 32.0), min_alpha=min_alpha,
            # floater 抑制参数（详见 mesh_export.filter_depth 的注释）
            erode_px=int(ctx.cfg("mesh_erode_px", 1) or 0),
            grad_thresh_ratio=float(ctx.cfg("mesh_grad_thresh", 0.06) or 0.0),
        )
        ctx.info(f"融合完成：用了 {info['views_used']} 个视角，体素 {info['voxel']:.5f}"
                 f"（设备 {info['device']}），深度过滤保留率 {info['filter_keep_ratio']*100:.1f}%，"
                 f"得到 {len(mesh.triangles)} 个三角形 / {len(mesh.vertices)} 个顶点")

        removed = 0
        if largest_only:
            mesh, removed = to_largest_component(mesh)
            ctx.info(f"只保留最大连通块，去掉 {removed} 个三角形")
        else:
            mesh, removed = clean_mesh(mesh)

        if len(mesh.triangles) == 0:
            ctx.warn("TSDF 没提取到任何三角形：可能是视角太少、深度无效，"
                     "或体素过大。可试 --set export.mesh_voxel_div=800")
            return []

        return self._finish_mesh(ctx, mesh, info, out_dir, cams=cams)

    def _load_bake_inputs(self, ctx: StageContext, cams):
        """纹理烘焙要用的原图与相机参数（与 cams 同序）。

        用**去畸变后的训练图**（04_train/data/images）而不是 C+leaned 图：
        它们的投影关系与 c2w/K 严格一致，采样不会错位。
        """
        import numpy as np
        from PIL import Image

        job_dir = Path(getattr(ctx, "job_dir", Path(".")))
        img_dir = job_dir / "04_train" / "data" / "images"
        images, c2ws, ks, whs = [], [], [], []
        for c2w, K, w, h, name in cams:
            p = img_dir / name
            if not p.exists():
                cands = list(img_dir.glob(Path(name).stem + ".*")) if img_dir.exists() else []
                if not cands:
                    continue
                p = cands[0]
            try:
                with Image.open(p) as im:
                    images.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
            except Exception:  # noqa: BLE001
                continue
            c2ws.append(np.asarray(c2w, dtype=np.float64))
            ks.append(np.asarray(K, dtype=np.float64))
            whs.append((int(w), int(h)))
        return images, np.stack(c2ws), np.stack(ks), whs

    def _finish_mesh(self, ctx: StageContext, mesh, info: dict, out_dir: Path,
                     cams=None) -> list[dict]:
        """网格后处理与落盘：去离群 -> 写 obj/ply/glb -> 可选纹理烘焙 -> 预览 -> 统计。

        cams：训练相机位姿列表 [(c2w, K, w, h, name)]，给了才做纹理烘焙
              （烘焙要从原图采样颜色）。
        """
        import numpy as np
        import open3d as o3d

        from ..gsdata.mesh_export import clean_mesh, to_largest_component

        largest_only = ctx.cfg("mesh_largest_only", True)
        if isinstance(largest_only, str):
            largest_only = largest_only.strip().lower() in ("1", "true", "yes", "on")
        removed = 0
        if largest_only:
            mesh, removed = to_largest_component(mesh)
            ctx.info(f"只保留最大连通块，去掉 {removed} 个三角形")
        else:
            mesh, removed = clean_mesh(mesh)

        if len(mesh.triangles) == 0:
            ctx.warn("网格后处理后为空，放弃导出")
            return []

        # ---- 法线统一朝外（必须在**任何写盘之前**）----
        #
        # 踩过的坑（两次）：
        #   1) 最初只在"烘焙前"定向 ⇒ 而写 OBJ/PLY/GLB 发生在它**之前**，
        #      于是导出文件的朝外比例仍是 85.4%（验收 FAIL）；
        #   2) 定向后再跑 QEM 简化/删面会打乱部分绕序。
        # 现在：后处理（简化/删面/最大连通块）**全部做完**，再定向，
        # 然后才写文件 —— 保证"落盘的东西"就是定向后的。
        # 实测：同网格补一次定向 85.1% → **100%**（翻转 7,270 面）。
        try:
            before = mesh
            mesh = _orient_normals_outward(mesh)
            if mesh is not before:
                ctx.info("法线统一朝外（写盘前）")
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"法线定向失败（{type(exc).__name__}: {exc}）")

        out: list[dict] = []
        obj_path = out_dir / "model_mesh.obj"
        o3d.io.write_triangle_mesh(str(obj_path), mesh, write_vertex_colors=True)
        out.append({"kind": "mesh_obj", "path": obj_path.name,
                    "size_mb": round(obj_path.stat().st_size / 1024 / 1024, 2),
                    "desc": (f"三角网格 OBJ（{len(mesh.triangles)} 面，"
                             f"{info.get('method', 'mesh')}，"
                             f"{info.get('views_used', info.get('points_used', 0))} 输入）")})
        ctx.success(f"已导出 {obj_path.name}（{len(mesh.triangles)} 个三角形）")

        ply_path = out_dir / "model_mesh.ply"
        o3d.io.write_triangle_mesh(str(ply_path), mesh, write_vertex_colors=True)
        out.append({"kind": "mesh_ply", "path": ply_path.name,
                    "size_mb": round(ply_path.stat().st_size / 1024 / 1024, 2),
                    "desc": "三角网格 PLY（与 OBJ 同源）"})

        # 网格 GLB：通用查看器 / Blender / Unity / three.js 都能直接打开带面的模型
        # （点云 glb 只是"一堆点"，没有实感，所以网格版是必要的产物）
        try:
            glb_path = self._write_mesh_glb(out_dir / "model_mesh.glb", mesh)
            vres = validate_glb(glb_path)
            if not vres["ok"]:
                ctx.warn("网格 GLB 规范校验未通过：" + "；".join(vres["problems"]))
            out.append({"kind": "mesh_glb", "path": glb_path.name,
                        "size_mb": round(glb_path.stat().st_size / 1024 / 1024, 2),
                        "valid": bool(vres["ok"]),
                        "desc": (f"三角网格 glTF 2.0（TRIANGLES 图元 + 顶点色，"
                                 f"{len(mesh.triangles)} 面），可直接进 Blender/Unity/three.js")})
            ctx.success(f"已导出 {glb_path.name}（网格 glb，校验{'通过' if vres['ok'] else '未通过'}）")
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"网格 GLB 导出失败（obj/ply 已导出）：{type(exc).__name__}: {exc}")

        # ---- 纹理烘焙（可选）：UV 展开 + 从原图烘焙漫反射贴图 ----
        # 顶点色在细节上远不如贴图（顶点数决定颜色分辨率），
        # 烘焙后模型能带"原图质感"，这是"实感"的最后一步。
        if cams and ctx.cfg("mesh_texture", False):
          try:
            from ..gsdata.mesh_export import bake_texture_from_views, unwrap_uv

            tex_size = int(ctx.cfg("mesh_texture_size", 2048) or 2048)
            # ⚠️ UV 展开是**单个 C 调用，无法中断**，而且实测耗时超线性：
            #     xatlas 在 9,940 面的合成球体上 0.61s，
            #     但在 79,808 面的泊松网格上要 **813 秒（13.5 分钟）** ——
            #     面数只多 8 倍，时间多 1330 倍；107,635 面时更久（实测 >20 分钟未完成）。
            # 后果不只是慢：manifest.json 原来写在所有步骤之后，
            # 一旦烘焙卡住/被中断，已导出的 glb/obj/ply 会变成**没有清单的孤儿产物**。
            # 对策：给 UV 展开加**超时**（放进线程，超时就跳过纹理烘焙、流程继续），
            # 让"纹理"成为可选的加分项，而不是阻塞整条导出的必经环节。
            timeout_s = float(ctx.cfg("mesh_texture_timeout", 300) or 0)
            n_tri_now = len(mesh.triangles)
            # ---- 烘焙前把法线统一成朝外（本轮修的 54% 纹理黑洞）----
            #
            # 实测 img3-crop：网格**只有 43.6% 的面法线朝外**（is_orientable=False、
            # 非封闭）。而 `bake_texture_from_views` 用
            #     facing = -(N·vd);  ok &= facing > 0.15
            # 挑选视角 —— 朝内的面**任何视角都被剔除**，对应 texel 填不上颜色，
            # 保持纯黑。实测 **53.96% 的 UV 覆盖 texel 是黑的** ⇒ 模型上大片黑斑。
            #
            # 修法：按"从物体中心向外"翻转绕序（实测 43.6% → **100.0%**）。
            # 注意**不能**依赖 `orient_triangles()`：对这份非封闭网格它是空操作
            # （以前实测 17.9% → 17.9%）。
            try:
                mesh = _orient_normals_outward(mesh)
                ctx.info("烘焙前法线定向：已按'中心向外'统一绕序")
            except Exception as exc:  # noqa: BLE001
                ctx.warn(f"法线定向失败（{type(exc).__name__}: {exc}），烘焙可能产生黑斑")
            ctx.info(f"⑦ 纹理烘焙：UV 展开（xatlas）+ 从 {len(cams)} 个视角采样 @{tex_size}²"
                     f"（网格 {n_tri_now} 面，超时 {timeout_s:.0f}s）...")
            uv_mesh, uv_tris, uv, uw_sec = _unwrap_uv_with_timeout(
                unwrap_uv, mesh, tex_size, timeout_s)
            if uv_mesh is None:
                ctx.warn(
                    f"UV 展开超时（>{timeout_s:.0f}s，网格 {n_tri_now} 面），**跳过纹理烘焙**；"
                    f"网格与点云产物不受影响。可降低面数"
                    f"（--set export.mesh_target_triangles=40000）、降低贴图分辨率，"
                    f"或调大 --set export.mesh_texture_timeout=1200")
            else:
                ctx.info(f"   UV 展开完成：{uw_sec:.1f}s")
                images, c2ws, ks, whs = self._load_bake_inputs(ctx, cams)
                tex, tinfo = bake_texture_from_views(
                    uv_mesh, uv_tris, uv, images, c2ws, ks, whs, tex_size=tex_size)
                ctx.info(f"   贴图覆盖 UV 图集 {tinfo['coverage']*100:.1f}%")

                o3d.io.write_triangle_mesh(str(out_dir / "model_mesh_uv.obj"), uv_mesh,
                                           write_vertex_colors=True, write_triangle_uvs=True)
                from PIL import Image as _Img

                _Img.fromarray(tex).save(out_dir / "model_mesh_texture.png")
                # 带贴图的 GLB：**必须自己拼**。open3d 写出带贴图网格时会报
                # "This file format does not support writing textures and uv
                # coordinates"，产出的 glb 里没有 images/textures/materials
                # （实测解析确认），贴图会被丢掉。
                tex_glb = out_dir / "model_mesh_textured.glb"
                self._write_mesh_glb(tex_glb, uv_mesh, name="textured_mesh",
                                     texture_png=(out_dir / "model_mesh_texture.png").read_bytes(),
                                     corner_uv=np.asarray(uv_mesh.triangle_uvs))
                out.append({"kind": "mesh_textured_glb", "path": tex_glb.name,
                            "size_mb": round(tex_glb.stat().st_size / 1024 / 1024, 2),
                            "coverage": round(tinfo["coverage"], 4),
                            "desc": f"带贴图的网格 glTF 2.0（{tex_size}² 贴图，UV 展开）"})
                out.append({"kind": "mesh_texture", "path": "model_mesh_texture.png",
                            "size_mb": round((out_dir / 'model_mesh_texture.png').stat().st_size / 1024 / 1024, 2),
                            "desc": "烘焙出的漫反射贴图"})
                out.append({"kind": "mesh_uv_obj", "path": "model_mesh_uv.obj",
                            "size_mb": round((out_dir / 'model_mesh_uv.obj').stat().st_size / 1024 / 1024, 2),
                            "desc": "带 UV 的网格 OBJ（配 model_mesh_texture.png 使用）"})
                ctx.success(f"已导出 {tex_glb.name}（带贴图网格）")
          except Exception as exc:  # noqa: BLE001
            ctx.warn(f"纹理烘焙失败（无贴图版本已导出）：{type(exc).__name__}: {exc}")

        bb = mesh.get_axis_aligned_bounding_box()
        out.append({"kind": "mesh_stats",
                    "method": info.get("method", "tsdf"),
                    "triangles": len(mesh.triangles), "vertices": len(mesh.vertices),
                    "bbox_extent": [round(float(v), 4) for v in bb.get_extent()],
                    "watertight": bool(mesh.is_edge_manifold()),
                    "views_used": info.get("views_used"),
                    "points_used": info.get("points_used"),
                    "voxel": (round(info["voxel"], 5) if "voxel" in info else None),
                    "desc": "网格统计（供质量判断）"})

        # 网格预览：否则用户拿到 obj 无从判断好坏（要开 Blender 才知道）
        try:
            from ..gsdata.mesh_export import render_mesh_previews

            prevs = render_mesh_previews(mesh, out_dir,
                                         size=int(ctx.cfg("mesh_preview_size", 512) or 512))
            if prevs:
                out.extend(prevs)
                ctx.info(f"已生成 {len(prevs)} 张网格预览图（离屏渲染）")
            else:
                ctx.warn("网格预览渲染返回空白图（本机 Filament 离屏渲染在无窗口环境不可用），"
                         "已跳过。网格文件本身正常，可用 Blender/MeshLab 打开查看；"
                         "质量可用 manifest 里的 mesh_stats 判断。")
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"网格预览渲染失败（网格本身已导出）：{type(exc).__name__}: {exc}")
        return out

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

    # -------------------------------------------------------------- 网格 GLB
    def _write_mesh_glb(self, path: Path, mesh, name: str = "mesh",
                        texture_png: bytes | None = None,
                        corner_uv: "np.ndarray | None" = None) -> Path:
        """把三角网格写成 glTF 2.0 二进制（`.glb`），**TRIANGLES 图元 + 顶点色**。

        给了 `texture_png` 与 `corner_uv` 时，额外写入 `TEXCOORD_0` 与内嵌贴图
        （`images`/`textures`/`samplers` + `baseColorTexture`）。

        ⚠️ 为什么自己写而不用 open3d 的写出：
        `o3d.io.write_triangle_mesh(..., write_triangle_uvs=True)` 遇到带贴图的
        网格会打印 "This file format does not support writing textures and uv
        coordinates"，产出的 .glb 里**没有 images/textures/materials**，贴图被丢掉
        （实测解析 JSON 只有 POSITION/COLOR_0/NORMAL）。所以带贴图的 glb 必须自己拼。

        坐标约定：glTF 是 **Y 轴向上**，而 COLMAP / 我们的重建是 Y 向下，
        所以这里对 y、z 取反让模型正立（与点云 GLB 的处理保持一致）。

        索引用 uint32（顶点可能上百万，uint16 会溢出）。
        """
        import numpy as np

        verts = np.asarray(mesh.vertices, dtype=np.float32).copy()
        tris = np.asarray(mesh.triangles, dtype=np.uint32)
        if len(verts) == 0 or len(tris) == 0:
            raise ValueError("网格为空，无法导出 glb")

        verts[:, 1] *= -1
        verts[:, 2] *= -1

        if mesh.has_vertex_colors():
            cols = np.clip(np.asarray(mesh.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
        else:
            cols = np.full((len(verts), 3), 200, dtype=np.uint8)

        if len(verts) > 0xFFFFFFFF:
            raise ValueError("顶点数超出 uint32 索引范围")

        # 三角形索引必须是**逆时针**（否则法线朝内、看起来是黑的）。
        # open3d 的 extract_triangle_mesh 一般已是 CCW；这里按面积符号统一校正一次。
        v = verts[tris]
        n_face = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        flip = (n_face * (v.mean(axis=1) + 1e-9)).sum(axis=1) < 0
        if flip.all():
            # ⚠️ `tris[:, ::-1]` 会产生 **int64** 视图，tobytes() 会写 8 字节/元素，
            # 而 glTF 声明的是 uint32 -> 查看器解析出天文数字索引。
            # 实测被 validate_glb 抓到："索引越界 max=3907240675 >= 顶点数 33024"。
            # 必须显式转回 uint32。
            tris = np.ascontiguousarray(tris[:, ::-1].astype(np.uint32))

        use_tex = texture_png is not None and corner_uv is not None
        if use_tex:
            # glTF 的 UV 原点在左上、v 向下；我们的 UV 已经是这个约定，直接用。
            uv = np.asarray(corner_uv, dtype=np.float32).reshape(-1, 2)
            if len(uv) != len(tris) * 3:
                raise ValueError(f"UV 角数 {len(uv)} != 面数×3 {len(tris)*3}")

        # 自检：索引必须落在顶点范围内。这里挡一道，比让查看器黑屏好。
        # （踩坑：`tris[:, ::-1]` 会产生 int64 视图，tobytes() 写 8 字节/元素，
        #   而 glTF 声明 uint32 -> 解析出天文数字索引，被 validate_glb 抓到。）
        tris = np.ascontiguousarray(tris, dtype=np.uint32)
        if int(tris.max(initial=0)) >= len(verts):
            raise ValueError(f"网格索引越界：max={int(tris.max())} >= 顶点数 {len(verts)}")

        pos_b = verts.tobytes()
        col_b = cols.tobytes()
        idx_b = tris.reshape(-1).tobytes()

        # ⚠️ **必须写 NORMAL 属性**（踩坑，直接对应"网格杂乱、尖锐"的观感）：
        # 早先这里只写了 POSITION + COLOR_0。glTF 规范下**没有 NORMAL 就是逐面平面着色**，
        # 于是每个三角形亮度单独跳变 —— 在查看器里看去就是"碎、乱、尖锐"，
        # 而模型本身其实没那么糟。加上顶点法线后查看器才能做平滑着色。
        # 顺带：mesh_export 里算过一致的顶点法线（orient_normals_consistent_tangent_plane），
        # 这里直接取用；取不到就现场算一次。
        try:
            nrm = np.asarray(mesh.vertex_normals, dtype=np.float32)
            if len(nrm) != len(verts):
                raise ValueError("法线数量与顶点不一致")
        except Exception:  # noqa: BLE001
            try:
                import open3d as _o3d

                m2 = _o3d.geometry.TriangleMesh()
                m2.vertices = _o3d.utility.Vector3dVector(verts.astype(np.float64))
                m2.triangles = _o3d.utility.Vector3iVector(tris.astype(np.int32))
                m2.compute_vertex_normals()
                nrm = np.asarray(m2.vertex_normals, dtype=np.float32)
            except Exception:  # noqa: BLE001
                nrm = np.zeros_like(verts, dtype=np.float32)
        # 导出时 y、z 已取反，法线必须同步取反，否则光照方向整体反掉
        nrm = np.ascontiguousarray(nrm)
        nrm[:, 1] *= -1
        nrm[:, 2] *= -1
        nrm_b = nrm.tobytes()

        assert len(idx_b) == tris.size * 4, (len(idx_b), tris.size)
        assert struct.unpack_from("<I", idx_b, 0)[0] < len(verts), \
            f"索引首元素异常：{struct.unpack_from('<I', idx_b, 0)[0]}"
        parts: list[bytes] = []
        cur = 0

        def _add(blob: bytes) -> tuple[int, int]:
            """按 4 字节对齐追加一段二进制，返回 (起始偏移, 未加 padding 的长度)。

            ⚠️ 踩坑（找了很久的一个低级错误）：最初写成 `parts = [pos_b]` 先把位置
            数据放进列表，然后又调 `_add(pos_b)` 追加了一次 ——
            等于**位置数据写了两遍**，索引整体后移，而各 bufferView 声明的 offset
            仍按"只写一遍"推导，于是"头部自洽、内容错位"：
            `validate_glb` 报 `索引越界 max=3223540644`（那其实是 float 的位模式）。
            教训：**所有段落一律走同一个 `_add`**，不要在初始化时就预置元素。
            """
            nonlocal cur
            pad = (-len(blob)) % 4
            start = cur
            parts.append(blob + b"\x00" * pad)
            cur += len(blob) + pad
            return start, len(blob)

        pos_off, pos_len = _add(pos_b)
        col_off, col_len = _add(col_b)
        nrm_off, nrm_len = _add(nrm_b)
        uv_off = uv_len = 0
        if use_tex:
            uv_off, uv_len = _add(uv.tobytes())
        idx_off, idx_len = _add(idx_b)
        img_off = img_len = 0
        if use_tex:
            img_off, img_len = _add(texture_png)
        blob = b"".join(parts)

        buffer_views = [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": col_off, "byteLength": col_len, "target": 34962},
            {"buffer": 0, "byteOffset": nrm_off, "byteLength": nrm_len, "target": 34962},
        ]
        accessors = [
            {"bufferView": 0, "componentType": COMPONENT_FLOAT, "count": int(len(verts)),
             "type": "VEC3",
             "min": [float(verts[:, i].min()) for i in range(3)],
             "max": [float(verts[:, i].max()) for i in range(3)]},
            {"bufferView": 1, "componentType": 5121, "normalized": True,
             "count": int(len(verts)), "type": "VEC3"},
            {"bufferView": 2, "componentType": COMPONENT_FLOAT,
             "count": int(len(verts)), "type": "VEC3"},
        ]
        attributes = {"POSITION": 0, "COLOR_0": 1, "NORMAL": 2}
        n_bv = 3
        idx_accessor = None
        if use_tex:
            buffer_views.append({"buffer": 0, "byteOffset": uv_off,
                                 "byteLength": uv_len, "target": 34962})
            accessors.append({"bufferView": n_bv, "componentType": COMPONENT_FLOAT,
                              "count": int(len(uv)), "type": "VEC2"})
            attributes["TEXCOORD_0"] = n_bv
            n_bv += 1
        buffer_views.append({"buffer": 0, "byteOffset": idx_off,
                             "byteLength": idx_len, "target": 34963})
        accessors.append({"bufferView": n_bv, "componentType": 5125,
                          "count": int(tris.size), "type": "SCALAR"})
        idx_accessor = n_bv
        n_bv += 1
        if use_tex:
            buffer_views.append({"buffer": 0, "byteOffset": img_off, "byteLength": img_len})

        material = {
            "name": "mesh_surface",
            "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                     "metallicFactor": 0.0, "roughnessFactor": 0.8},
            "doubleSided": True,
        }
        gltf: dict = {
            "asset": {"version": "2.0", "generator": "gsflow"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": name}],
            "meshes": [{"name": name, "primitives": [{
                "attributes": attributes, "indices": idx_accessor,
                "mode": MODE_TRIANGLES, "material": 0}]}],
            "buffers": [{"byteLength": len(blob)}],
            "bufferViews": buffer_views,
            "accessors": accessors,
            "extras": {"triangles": int(len(tris)), "vertices": int(len(verts)),
                       "textured": bool(use_tex), "source": "gsflow-mesh"},
        }
        if use_tex:
            gltf["materials"] = [material]
            gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
            gltf["textures"] = [{"source": 0, "sampler": 0}]
            gltf["images"] = [{"bufferView": n_bv, "mimeType": "image/png"}]
            gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987,
                                 "wrapS": 10497, "wrapT": 10497}]
        else:
            gltf["materials"] = [material]

        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_bytes += b" " * ((-len(json_bytes)) % 4)

        # 写盘前自检：直接从**最终 blob**里按 accessor 读回索引并与内存比对。
        # 结构与内容都自洽才允许落盘（这样就不会再把"声明对、内容错"的文件交出去）。
        _bv = buffer_views[idx_accessor]["byteOffset"]
        _back = np.frombuffer(blob[_bv:_bv + 16], dtype="<u4")
        assert np.array_equal(_back, tris.reshape(-1)[:len(_back)]), (
            "索引回读不一致：blob=" + str(_back.tolist())
            + " vs tris=" + str(tris.reshape(-1)[:len(_back)].tolist()))
        assert int(_back.max()) < len(verts), f"回读索引越界：{int(_back.max())} >= {len(verts)}"

        total = 12 + 8 + len(json_bytes) + 8 + len(blob)
        with path.open("wb") as fh:
            fh.write(struct.pack("<III", GLB_MAGIC, 2, total))
            fh.write(struct.pack("<II", len(json_bytes), JSON_CHUNK))
            fh.write(json_bytes)
            fh.write(struct.pack("<II", len(blob), BIN_CHUNK))
            fh.write(blob)
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
    def _splat_kernel_ready(self) -> bool:
        """gsplat 的 CUDA 内核是否**已就绪且不会触发编译**。

        为什么要单独判断：`import gsplat` 之后第一次调用 `rasterization` 可能**现场
        JIT 编译**内核（本机实测 80~140 秒）。预览图只是"顺手看一眼"的产物，
        绝不该因为要出预览而触发一次重量级编译 ——
        在测试里表现为单个用例从 5 秒变成 86 秒（实测踩到）。

        判据：`paths.apply_runtime_env()` 是否已把预编译内核注册进 `sys.modules`
        （它会同时注册 `gsplat_cuda` 与 `gsplat.csrc`，从而让 gsplat 跳过编译分支）。
        这个判据与实际机制一致 —— 早期我写成"看扩展目录里有没有 `.pyd`"，
        结果在没预加载的进程里**误判为就绪**，照样触发了 86 秒编译。
        """
        import sys as _sys

        if _sys.modules.get("gsplat.csrc") is None:
            return False
        try:
            import gsplat  # noqa: F401

            from gsplat.cuda._backend import _C

            return _C is not None
        except Exception:  # noqa: BLE001
            return False

    def _render_previews(self, ctx: StageContext, gauss: dict, out_dir: Path, count: int = 4) -> list[dict]:
        """生成预览图：优先用 **gsplat 真实渲染**，失败时退回 matplotlib 散点投影。

        为什么要真渲染：matplotlib 版只是把高斯**中心点**画成散点，
        得到的是一团点云，看不出重建质量（一个 PSNR 35dB 的好模型和一堆乱点长得一样）。
        真渲染能直接反映"模型看到的世界"，也是用户最关心的产物。

        gsplat 内核未就绪或不可用时自动退回散点图，保证导出阶段仍能独立跑通，
        且**不会**为了出预览而触发一次 80 秒级的内核编译。
        """
        if not self._splat_kernel_ready():
            ctx.info("gsplat CUDA 内核尚未构建，预览改用散点投影（避免触发 JIT 编译）")
            return self._render_previews_scatter(ctx, gauss, out_dir, count)
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
            # 预览也必须用**完整 SH** 渲染，否则预览图不是模型真实外观
            shs_full, sh_deg = _sh_tensors(gauss, device="cuda")

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
            shs_full = shs_full[keep]
            n = int(xyz.shape[0])
            ctx.info(f"预览渲染使用 {n} 个可见高斯（不透明度 > 0.02），SH 阶数 {sh_deg}")

            # 首选：直接用**训练时的真实相机位姿**渲染。
            # 自己估的"环绕相机"很容易取景失当（实测把画面渲成一片背景色），
            # 而训练相机的视角天然合理，也正好能反映"模型在训练视图上看到什么"。
            poses = self._training_cameras(ctx, out_dir)
            if poses:
                return self._render_from_poses(ctx, xyz, scales_real, quats, opac, rgb,
                                               poses, out_dir, count,
                                               shs=shs_full, sh_degree=sh_deg)

            center = xyz.mean(dim=0)
            spread = torch.linalg.norm(xyz - center, dim=1) + scales_real.max(dim=1).values
            radius = float(spread.max()) * 2.6 + 1e-3
            H = W = 512
            fov = 1.0
            focal = 0.5 * W / np.tan(0.5 * fov)
            shs = shs_full

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
                        width=W, height=H, sh_degree=int(sh_deg), packed=False,
                        render_mode="RGB",
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

    def _training_cameras(self, ctx: StageContext, out_dir: Path, limit: int = 4):
        """读取训练阶段用的 COLMAP 稀疏模型，取若干真实相机位姿（c2w）与 K。

        limit：最多返回多少个位姿。**预览**只需要 4 个不同视角即可；
        但 **TSDF 网格融合**需要尽量多的视角覆盖物体表面，所以调用方要传大值
        （踩坑：早期这个方法硬抽 4 个，网格导出只有 4 个视角可用，融合质量极差）。

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
        # 均匀抽取，避免都挤在同一侧
        if limit and len(poses) > limit:
            idx = np.linspace(0, len(poses) - 1, limit).round().astype(int)
            poses = [poses[i] for i in idx]
        ctx.info(f"读取到 {len(poses)} 个训练相机位姿（来自 {model_dir.name}）")
        return poses

    def _render_from_poses(self, ctx: StageContext, xyz, scales_real, quats, opac, rgb,
                           poses, out_dir: Path, count: int,
                           shs=None, sh_degree: int = 0) -> list[dict]:
        """用给定的真实相机位姿渲染预览（gt 图若存在则并排拼上，便于对照）。

        注意 scales_real 是**真实尺度**（_read_ply 已指数化），不要再 exp。
        `shs` 若传入则为**完整 SH**（含高阶），预览才是模型真实外观；
        未传入时回退为仅 DC（老 PLY）。
        """
        import torch
        from gsplat import rasterization
        from PIL import Image

        n = int(xyz.shape[0])
        if shs is None:
            shs = ((rgb - 0.5) / 0.28209479177387814).reshape(n, 1, 3)
            sh_degree = 0
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
                    width=W, height=H, sh_degree=int(sh_degree), packed=False,
                    render_mode="RGB",
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
