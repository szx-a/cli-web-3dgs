"""导出 `.splat`（32 字节/高斯）—— 让查看器能渲染**真正的泼溅**而不是点云。

## 为什么要它（用户反馈的直接原因）

用户说"GUI 查看器里的模型比 `preview_front.png` 差很多"。查清了：
* `preview_*.png` 是 **gsplat 泼溅渲染**（位置 + 3×3 协方差 + 不透明度 + SH3）；
* `viewer.html` / GUI 解析的是 **glTF POINTS / PLY 点云**，
  每个高斯被压成 `x,y,z,r,g,b` 六个参数 —— **丢了协方差、不透明度、SH**。
⇒ 点云必然是"稀疏点阵、有空隙、颜色扁平"，等价于把 59 参数的模型降维成 6 参数。

`.splat` 格式（antimatter15 提出，three.js / SuperSplat / gsplat.js 生态通用）
每条 **32 字节**，正好保留"位置 + 尺度 + 不透明度 + 旋转"，查看器据此可以：
  * 把每个高斯画成**朝向各异的椭球**（而不是一个点）；
  * 做 **alpha 合成**（而不是硬点）；
  * 得到连续表面。

布局（小端）：

| 偏移 | 类型 | 含义 |
|---|---|---|
| 0  | 3 × float32 | 位置 x, y, z |
| 12 | 3 × float32 | 尺度 sx, sy, sz（**已取指数**，世界单位）|
| 24 | 4 × uint8   | 颜色 R, G, B, A（A = 不透明度 × 255）|
| 28 | 4 × uint8   | 旋转四元数 w, x, y, z（各 = q×128+128）|

⚠️ 四元数用 **w 在前**（该格式的约定），而我们的 PLY 里是 `rot_0..rot_3` = **w,x,y,z**，
所以直接顺序写即可（这一点必须核对，写反会让椭球朝向全错）。

不依赖任何外部包（`spz` 包未安装时 `_write_spz` 会静默返回 None）。
"""
from __future__ import annotations

import numpy as np

SPLAT_BYTES = 32


def gaussians_to_splat(gauss: dict) -> bytes:
    """把 `read_gaussians_full` 的结果打包成 `.splat` 字节流。"""
    xyz = np.asarray(gauss["xyz"], dtype=np.float32).reshape(-1, 3)
    scales = np.asarray(gauss["scales"], dtype=np.float32).reshape(-1, 3)
    quats = np.asarray(gauss["quats"], dtype=np.float32).reshape(-1, 4)
    opac = np.asarray(gauss["opacity"], dtype=np.float32).reshape(-1)
    rgb = np.asarray(gauss["rgb"], dtype=np.float32).reshape(-1, 3)

    n = len(xyz)
    if n == 0:
        return b""
    if not (len(scales) == len(quats) == len(opac) == len(rgb) == n):
        raise ValueError(f"高斯属性数量不一致：xyz={n} scales={len(scales)} "
                         f"quats={len(quats)} opac={len(opac)} rgb={len(rgb)}")

    # 归一化四元数（格式要求单位四元数；未归一化会让椭球拉伸）
    qn = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(qn, 1e-9)

    # 颜色 0~255 已经就绪（read_gaussians_full 返回 0~255）
    rgba = np.empty((n, 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[:, 3] = np.clip(opac * 255.0, 0, 255).astype(np.uint8)

    # 四元数编码：byte = q*128 + 128，落在 0~255
    qb = np.clip(quats * 128.0 + 128.0, 0, 255).astype(np.uint8)

    out = np.empty(n * SPLAT_BYTES, dtype=np.uint8)
    out = out.reshape(n, SPLAT_BYTES)
    out[:, 0:12] = xyz.astype("<f4").view(np.uint8).reshape(n, 12)
    out[:, 12:24] = scales.astype("<f4").view(np.uint8).reshape(n, 12)
    out[:, 24:28] = rgba
    out[:, 28:32] = qb
    return out.tobytes()


def parse_splat_header(data: bytes) -> dict:
    """解析 `.splat` 头几个高斯，用于**验收**（校验字节布局是否写对）。"""
    n = len(data) // SPLAT_BYTES
    if n == 0:
        return {"count": 0}
    a = np.frombuffer(data[: min(len(data), SPLAT_BYTES * 4)], dtype=np.uint8)
    a = a.reshape(-1, SPLAT_BYTES)
    pos = a[:, 0:12].copy().view("<f4").reshape(-1, 3)
    scl = a[:, 12:24].copy().view("<f4").reshape(-1, 3)
    return {
        "count": n,
        "first_positions": pos.tolist(),
        "first_scales": scl.tolist(),
        "first_rgba": a[:, 24:28].tolist(),
        "first_quat_bytes": a[:, 28:32].tolist(),
    }
