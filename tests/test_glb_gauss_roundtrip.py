"""`glb_gauss` 的往返测试：写出来的 glb 必须**完整**还原 3DGS 参数。

背景（用户 2026-10-31 指出）：`model.glb` 只有几 MB 不是"压缩得好"，而是**丢了质量** ——
旧实现只写 `POSITION + COLOR_0`，尺度/旋转/不透明度/48 个球谐系数全部丢失，
于是渲染出来只能是稀疏点阵。

这个测试用 6 个合成高斯写一个 glb，再按 glTF 规范解析回来，逐字段比对：
位置（含 Y-up 翻转）、尺度、旋转、不透明度、SH 残差、基色，以及扩展声明。
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("plyfile")

from gsflow.stages.export import ExportStage  # noqa: E402


def _make_gauss(n: int = 6, sh_degree: int = 3):
    rng = np.random.default_rng(0)
    k = (sh_degree + 1) ** 2  # 16
    xyz = rng.normal(size=(n, 3)).astype(np.float32)
    scales = rng.normal(size=(n, 3)).astype(np.float32) * 0.05
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opac = rng.random((n, 1)).astype(np.float32)
    shs = rng.normal(size=(n, k, 3)).astype(np.float32) * 0.3
    return {"xyz": xyz, "scales": scales, "quats": quats, "opacity": opac,
            "shs": shs, "n": n, "sh_degree": sh_degree,
            "rgb": np.clip((shs[:, 0, :] * 0.28209479177387814 + 0.5) * 255, 0, 255).astype(np.uint8)}


def _read_glb(path):
    raw = path.read_bytes()
    magic, ver, total = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67 and ver == 2
    json_len, json_type = struct.unpack("<II", raw[12:20])
    assert json_type == 0x4E4F534A
    gltf = json.loads(raw[20:20 + json_len].decode("utf-8"))
    off = 20 + json_len
    bin_len, bin_type = struct.unpack("<II", raw[off:off + 8])
    assert bin_type == 0x004E4942
    blob = raw[off + 8:off + 8 + bin_len]
    return gltf, blob


def _accessor(gltf, blob, idx):
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    ncomp = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc["type"]]
    comp = acc["componentType"]
    dtype = {5126: "<f4", 5121: "u1"}[comp]
    size = np.dtype(dtype).itemsize
    start = bv["byteOffset"]
    n = acc["count"] * ncomp
    arr = np.frombuffer(blob, dtype=dtype, count=n, offset=start).reshape(acc["count"], ncomp)
    return arr


def test_glb_gauss_roundtrip(workdir) -> None:
    gauss = _make_gauss()
    out = workdir / "model_gauss.glb"
    ExportStage()._write_glb_gaussians(out, gauss)

    gltf, blob = _read_glb(out)
    assert "GSFLOW_gaussian_splatting" in gltf["extensionsUsed"]
    ext = gltf["extensions"]["GSFLOW_gaussian_splatting"]
    assert ext["gaussianCount"] == gauss["n"]
    assert ext["shDegree"] == 3

    attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
    pos = _accessor(gltf, blob, attrs["POSITION"])
    scale = _accessor(gltf, blob, attrs["_GS_SCALE"])
    rot = _accessor(gltf, blob, attrs["_GS_ROTATION"])
    opac = _accessor(gltf, blob, attrs["_GS_OPACITY"]).reshape(-1)

    # 位置：写的时候做了 glTF 的 Y-up 翻转，这里翻回来比较
    pos_back = pos.copy()
    pos_back[:, 1] *= -1.0
    pos_back[:, 2] *= -1.0
    assert np.allclose(pos_back, gauss["xyz"], atol=1e-5)
    assert np.allclose(scale, gauss["scales"], atol=1e-6)
    assert np.allclose(rot, gauss["quats"], atol=1e-6)
    assert np.allclose(opac, gauss["opacity"].reshape(-1), atol=1e-6)

    # SH 残差：48 个值（16-1 个球谐 × 3 通道）分散在 12 个 VEC4 里
    n_vec4 = ext["shRestVec4"]
    assert n_vec4 == 12
    parts = [_accessor(gltf, blob, attrs[f"_GS_SH_{i:02d}"]) for i in range(n_vec4)]
    flat = np.concatenate(parts, axis=1)          # [N, 48]
    expect = gauss["shs"][:, 1:, :].reshape(gauss["n"], -1)
    assert np.allclose(flat[:, :expect.shape[1]], expect, atol=1e-6)


def test_glb_gauss_is_not_the_lossy_points_glb(workdir) -> None:
    """完整 glb 必须**明显大于**只存位置+颜色的那个，并且带 SH 属性。"""
    gauss = _make_gauss(n=200)
    stage = ExportStage()
    full = workdir / "full.glb"
    stage._write_glb_gaussians(full, gauss)
    small = workdir / "small.glb"
    stage._write_glb(small, gauss)
    assert full.stat().st_size > small.stat().st_size * 5, (
        f"完整 glb 应远大于点云 glb：full={full.stat().st_size} small={small.stat().st_size}")
    gltf, _ = _read_glb(full)
    attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
    assert any(k.startswith("_GS_SH_") for k in attrs), "缺少 SH 属性"
    assert "_GS_SCALE" in attrs and "_GS_ROTATION" in attrs and "_GS_OPACITY" in attrs
