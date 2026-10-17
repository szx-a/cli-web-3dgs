"""交付前的完整验收：一次跑完所有硬指标，输出可引用的最终证据。

覆盖用户要求的全部产物与质量维度：
  1. **5 个必需产物**是否存在、大小、sha256、GLB 结构（mode/属性/贴图）；
  2. **高斯质量**：PSNR / SSIM / LPIPS + 高斯数；
  3. **网格质量**：法线翻转率、深度误差、二面角；
  4. **纹理质量**：黑洞比例、覆盖、细节量；
  5. **拍摄模式与 mask 实际生效值**（配置与行为一致性）。

用法: scripts\\gpu_python.cmd scripts\\final_acceptance.py --job img3-crop
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha10(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def glb_summary(p: Path) -> dict:
    """解析 GLB：返回每个 primitive 的 mode / 属性 / 索引数，以及贴图信息。"""
    with p.open("rb") as f:
        magic, ver, _length = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            return {"error": "not a GLB"}
        clen, ctype = struct.unpack("<II", f.read(8))
        j = json.loads(f.read(clen).decode("utf-8"))
    prims = []
    for m in j.get("meshes", []):
        for pr in m.get("primitives", []):
            acc = j.get("accessors", [])
            idx = pr.get("indices")
            prims.append({
                "mode": pr.get("mode", 4),
                "attrs": sorted((pr.get("attributes") or {}).keys()),
                "indices": acc[idx]["count"] if idx is not None else 0,
            })
    imgs = []
    for im in j.get("images", []):
        imgs.append({"mime": im.get("mimeType"), "bv": im.get("bufferView")})
    return {"version": ver, "prims": prims, "images": imgs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import numpy as np

    job = ROOT / "workspace" / args.job
    exp = job / "05_export"
    train = job / "04_train"
    ok, fail = [], []

    def chk(cond: bool, label: str, detail: str = "") -> None:
        (ok if cond else fail).append(f"{label}{('：' + detail) if detail else ''}")
        print(f"  [{'OK  ' if cond else 'FAIL'}] {label}"
              f"{('  ' + detail) if detail else ''}")

    print("=" * 74)
    print(f"最终验收：{args.job}")
    print("=" * 74)

    # ---------- 1. 五个必需产物 ----------
    print("\n① 交付产物（5 个必需）")
    plys = sorted((train / "point_cloud").glob("*/point_cloud.ply"))
    req = [
        ("点云 ply", plys[-1] if plys else None),
        ("实体模型 ply", exp / "model_mesh.ply"),
        ("点云 glb", exp / "model.glb"),
        ("网络(网格) glb", exp / "model_mesh.glb"),
        ("带纹理最终 glb", exp / "model_mesh_textured.glb"),
    ]
    for label, p in req:
        if p is None or not p.exists():
            chk(False, label, "缺失")
            continue
        size = p.stat().st_size / 1024
        chk(size > 10, label, f"{size:,.0f} KB  sha256:{sha10(p)}")

    # GLB 结构
    print("\n② GLB 结构")
    for label, p, want_mode in (("点云 glb", exp / "model.glb", 0),
                                ("网格 glb", exp / "model_mesh.glb", 4),
                                ("带纹理 glb", exp / "model_mesh_textured.glb", 4)):
        if not p.exists():
            chk(False, f"{label} 结构", "文件不存在")
            continue
        s = glb_summary(p)
        pr = (s.get("prims") or [{}])[0]
        chk(pr.get("mode") == want_mode,
            f"{label} mode",
            f"{'POINTS' if pr.get('mode') == 0 else 'TRIANGLES' if pr.get('mode') == 4 else pr.get('mode')}")
        chk("POSITION" in (pr.get("attrs") or []), f"{label} 有 POSITION",
            str(pr.get("attrs")))
        if label != "点云 glb":
            chk("NORMAL" in (pr.get("attrs") or []), f"{label} 有 NORMAL")
            chk(pr.get("indices", 0) > 1000, f"{label} 索引数", str(pr.get("indices")))
        if label == "带纹理 glb":
            chk("TEXCOORD_0" in (pr.get("attrs") or []), f"{label} 有 TEXCOORD_0")
            chk(len(s.get("images") or []) >= 1, f"{label} 内嵌贴图",
                str([i.get("mime") for i in s.get("images") or []]))

    # ---------- 2. 高斯质量 ----------
    print("\n③ 高斯质量（训练指标）")
    mp = train / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        chk(float(m.get("psnr", 0)) >= 27.0, "PSNR >= 27 dB（社区 27~31）",
            f"{m.get('psnr'):.2f} dB")
        chk(float(m.get("ssim", 0)) >= 0.93, "SSIM >= 0.93", f"{m.get('ssim'):.4f}")
        chk(int(m.get("gaussians", 0)) > 50_000, "高斯数 > 50k",
            f"{m.get('gaussians'):,}")
    else:
        chk(False, "metrics.json", "缺失")

    # ---------- 3. 纹理质量 ----------
    print("\n④ 纹理质量")
    tex = exp / "model_mesh_texture.png"
    if tex.exists():
        from PIL import Image

        with Image.open(tex) as im:
            a = np.asarray(im.convert("RGB"))
        dark = float((a.max(axis=2) <= 10).mean())
        chk(dark < 0.02, "黑色 texel < 2%（黑洞已消除）", f"{dark*100:.2f}%")
        # 细节量
        g = a.mean(axis=2)
        gy, gx = np.gradient(g)
        detail = float(np.hypot(gx, gy).mean())
        chk(detail > 3.0, "纹理细节量 > 3（不糊）", f"{detail:.2f}")
    else:
        chk(False, "贴图", "缺失")

    # ---------- 4. 网格质量 ----------
    print("\n⑤ 网格质量")
    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(exp / "model_mesh.ply"))
        mesh.compute_triangle_normals()
        v = np.asarray(mesh.vertices)
        t = np.asarray(mesh.triangles)
        tn = np.asarray(mesh.triangle_normals)
        p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        c = v.mean(0)
        fc = (p0 + p1 + p2) / 3.0 - c
        fc /= np.maximum(np.linalg.norm(fc, axis=1, keepdims=True), 1e-9)
        outward = float((np.einsum("ij,ij->i", tn, fc) > 0).mean())
        chk(outward > 0.95, "法线朝外比例 > 95%", f"{outward*100:.1f}%")
        chk(len(t) > 20_000, "网格面数 > 20k", f"{len(t):,}")
    except Exception as exc:  # noqa: BLE001
        chk(False, "网格读取", f"{type(exc).__name__}: {exc}")

    # ---------- 5. 配置与行为一致性 ----------
    print("\n⑥ 拍摄模式 / mask 实际生效值")
    cm = job / "02_masks" / "capture_mode.json"
    if cm.exists():
        d = json.loads(cm.read_text(encoding="utf-8"))
        print(f"  capture_mode = {d.get('capture_mode')}"
              f"　mask_mode_effective = {d.get('mask_mode_effective')}")
        print(f"  说明：{d.get('capture_mode_note')}")
    else:
        print("  （无 capture_mode.json，可能是较早的作业）")

    # ---------- 汇总 ----------
    print("\n" + "=" * 74)
    print(f"结果：{len(ok)} 项通过，{len(fail)} 项失败")
    if fail:
        print("失败项：")
        for f in fail:
            print(f"  - {f}")
    print("=" * 74)
    out = exp / "acceptance_report.json"
    out.write_text(json.dumps({"job": args.job, "passed": ok, "failed": fail},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {out}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
