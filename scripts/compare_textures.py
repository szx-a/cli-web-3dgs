"""对比两张贴图：黑色 texel 比例、颜色差异分布、以及"跨视角一致性"代理指标。

注意：被遮挡拒掉的 texel 会被**扩散填充**从邻居补色，所以不能只看颜色差异；
这里给出三个可解释的量：
1. `black_ratio`  纯黑 texel 比例（越低越好，说明没漏采）；
2. `changed_ratio` 两版差异 > 32/255 的 texel 比例（说明遮挡确实改变了采样）；
3. `edge_sharpness` 贴图内的梯度能量（杂糅会显著降低结构清晰度）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

A = Path(sys.argv[1] if len(sys.argv) > 1 else r"F:\3dgs-and-ai\env\tmp\tex_no_occlusion.png")
B = Path(sys.argv[2] if len(sys.argv) > 2 else r"F:\3dgs-and-ai\workspace\img5-solid-v3\05_export\model_mesh_texture.png")


def load(p: Path) -> np.ndarray:
    im = Image.open(p).convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


a, b = load(A), load(B)
print(f"  A（无遮挡）: {A.name}  {a.shape}")
print(f"  B（有遮挡）: {B.name}  {b.shape}")
if a.shape != b.shape:
    print("  尺寸不同，先 resize B 到 A")
    b = np.asarray(Image.open(B).convert("RGB").resize((a.shape[1], a.shape[0])), dtype=np.float32) / 255.0

def black_ratio(x: np.ndarray) -> float:
    return float((x.max(axis=2) < 6 / 255.0).mean())

diff = np.abs(a - b).max(axis=2)
changed = float((diff > 32 / 255.0).mean())

def grad_energy(x: np.ndarray) -> float:
    g = x.mean(axis=2)
    dx = np.abs(np.diff(g, axis=1)).mean()
    dy = np.abs(np.diff(g, axis=0)).mean()
    return float(dx + dy)

print(f"  黑色 texel 比例: A={black_ratio(a)*100:.2f}%  B={black_ratio(b)*100:.2f}%")
print(f"  两版差异>32/255 的 texel: {changed*100:.2f}%")
print(f"  梯度能量: A={grad_energy(a):.5f}  B={grad_energy(b):.5f}  （越高结构越清楚）")

# 保存可视化：差异热图 + 并排裁剪
out = Path(r"F:\3dgs-and-ai\env\tmp")
Image.fromarray((np.clip(diff, 0, 1) * 255).astype(np.uint8)).save(out / "tex_diff.png")
h, w = a.shape[:2]
cy, cx = h // 2, w // 2
crop = 256
box = (max(0, cy - crop), max(0, cx - crop), min(h, cy + crop), min(w, cx + crop))
side = np.concatenate([a[box[0]:box[2], box[1]:box[3]], b[box[0]:box[2], box[1]:box[3]]], axis=1)
Image.fromarray((side * 255).astype(np.uint8)).save(out / "tex_side_by_side.png")
print(f"  已写 {out / 'tex_diff.png'} 与 {out / 'tex_side_by_side.png'}")
