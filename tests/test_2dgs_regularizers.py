"""2DGS 正则的数学正确性测试（纯 CPU，不依赖 GPU）。

覆盖三件容易写错、且错了会"看起来加了正则其实什么都没做"的事：

1. `gaussian_normals` 必须取**最短轴**、且是单位向量。
2. `depth_to_normal` 必须给出**朝相机**的法线，并且对已知平面给出正确方向。
3. `two_dgs_regularizers` 的**深度失真项**必须是真损失：
   深度图越不平滑，损失越大；且**不能恒为 0**。

第 3 条尤其重要 —— 第一版我把法线一致性写成了恒等式（恒为 0），
如果当时有这条测试就会立刻暴露。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")


def test_gaussian_normals_uses_rotated_z_axis() -> None:
    """高斯法线取**旋转后的 +z 轴**（圆盘法线），而不是 `argmin(scales)` 的最短轴索引。

    为什么改：实测（scripts/compare_normal_sources.py，对压平过的真实模型比较邻域一致度）

        +z 轴             mean_dot 0.9569   翻转 0.0%
        最短轴 argmin      mean_dot 0.5690   翻转 4.4%
        最长轴             mean_dot 0.5586   翻转 5.1%

    而且"最短轴恰好是 z 轴"只在 **19.2%** 的高斯上成立 —— 旧实现大部分时候
    在监督一个水平方向，正则因而基本无效。这里用"最短轴是 x"的构造来锁住这个区别：
    正确实现应仍返回 x-y 平面内的 **z 轴方向**（由四元数决定），而不是 x 轴。
    """
    from gsflow.trainers.gsplat_trainer import gaussian_normals

    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])       # 无旋转 ⇒ 三个轴就是 x/y/z
    s = torch.tensor([[1.0, 1.0, 0.01]])
    n = gaussian_normals(q, s)
    assert n.shape == (1, 3)
    assert torch.allclose(n.norm(dim=-1), torch.ones(1), atol=1e-6), "法线必须单位化"
    assert abs(float(n[0, 2])) > 0.99, f"应取 +z 轴，实际 {n.tolist()}"

    # 关键区分：最短轴是 x —— 正确实现仍应返回 +z，而不是 x
    s2 = torch.tensor([[0.01, 1.0, 1.0]])
    n2 = gaussian_normals(q, s2)
    assert abs(float(n2[0, 2])) > 0.99, (
        f"最短轴是 x 时仍须返回 +z（圆盘法线），实际 {n2.tolist()} —— "
        "若返回 x 说明又回到了 argmin 索引的旧实现")

    # 绕 X 轴转 90°：+z 轴应转到 -y
    q90 = torch.tensor([[0.70710678, 0.70710678, 0.0, 0.0]])
    n3 = gaussian_normals(q90, s)
    assert abs(float(n3[0, 1])) > 0.99, f"绕 X 转 90° 后 +z 应为 ±y，实际 {n3.tolist()}"

    # axis='shortest' 仍可用于实验场景
    n4 = gaussian_normals(q, s2, axis="shortest")
    assert abs(float(n4[0, 0])) > 0.99, f"axis='shortest' 应取最短轴 x，实际 {n4.tolist()}"


def test_depth_to_normal_points_towards_camera() -> None:
    """正对相机的平面：法线应指向相机（相机系下 -Z 方向）。"""
    from gsflow.trainers.gsplat_trainer import depth_to_normal

    H = W = 32
    K = torch.tensor([[50.0, 0.0, W / 2.0], [0.0, 50.0, H / 2.0], [0.0, 0.0, 1.0]])
    # 常数深度 = 正对相机的平面
    depth = torch.full((H, W), 2.0)
    n, valid = depth_to_normal(depth, K)
    # 常数深度处一阶微分为 0 -> 切向量退化，valid 可能为 False；
    # 用**倾斜平面**来检验方向更可靠
    ys = torch.arange(H, dtype=torch.float32).view(-1, 1).expand(H, W)
    depth2 = 2.0 + 0.01 * ys          # 沿 v 方向线性变化 = 倾斜平面
    n2, valid2 = depth_to_normal(depth2, K)
    assert bool(valid2[1:-1, 1:-1].all()), "倾斜平面内部应全部有效"
    nn = n2[1:-1, 1:-1]
    assert torch.allclose(nn.norm(dim=-1), torch.ones_like(nn[..., 0]), atol=1e-5)
    # 统一朝向相机：与 -Z 的点积应 > 0
    dot = (nn * torch.tensor([0.0, 0.0, -1.0])).sum(-1)
    assert float(dot.min()) > 0.0, f"法线未朝向相机：min dot={float(dot.min()):.3f}"


def test_depth_to_normal_slope_direction() -> None:
    """深度随 v 增大而增大（平面向下远离相机）时，法线应向上倾斜。

    注意断言符号：深度沿 +v 增大 => 该方向远离相机 => 法线朝 -v（图像上方）。
    图像 v 轴向下，而相机系 +Y 向上，所以法线的 **Y 分量为正**。
    """
    from gsflow.trainers.gsplat_trainer import depth_to_normal

    H = W = 32
    K = torch.tensor([[50.0, 0.0, W / 2.0], [0.0, 50.0, H / 2.0], [0.0, 0.0, 1.0]])
    ys = torch.arange(H, dtype=torch.float32).view(-1, 1).expand(H, W)
    depth = 2.0 + 0.05 * ys
    n, valid = depth_to_normal(depth, K)
    assert bool(valid[1:-1, 1:-1].all())
    mid = n[H // 2, W // 2]
    assert float(mid[1]) > 0.0, f"法线 Y 分量应为正（朝上），实际 {mid.tolist()}"
    # 同时应朝向相机（-Z 分量占主导）
    assert float(mid[2]) < 0.0


def test_depth_to_normal_flat_plane_is_not_invalid() -> None:
    """**正对相机的平面**（深度为常数）必须给出有效法线。

    回归：梯度为零 => 叉积退化零向量。第一版把这种情况标记为"无效"，
    于是法线正则在最常见的"正对相机的大平面"上整片不生效。
    现在退回视线方向作法线（与导出阶段的兜底一致）。
    """
    from gsflow.trainers.gsplat_trainer import depth_to_normal

    H = W = 24
    K = torch.tensor([[50.0, 0.0, W / 2.0], [0.0, 50.0, H / 2.0], [0.0, 0.0, 1.0]])
    depth = torch.full((H, W), 2.0)
    n, valid = depth_to_normal(depth, K)
    assert bool(valid.all()), "常数深度图不应被整体判为无效"
    assert torch.allclose(n.norm(dim=-1), torch.ones_like(n[..., 0]), atol=1e-5)
    # ✅ 正确公式下，**正对相机的平面**法线应精确等于 [0,0,-1]。
    # 这条断言就是抓"切向量推导写错"的哨兵：错误公式会给出 [0,0.447,-0.894]
    # （偏离真实方向 26.6°），而它在**导出阶段**是同一个错误（已一并修正）。
    center = n[H // 2, W // 2]
    assert abs(float(center[2]) - (-1.0)) < 1e-5, (
        f"正面平行平面的法线应变为 [0,0,-1]，实际 {center.tolist()}")
    assert abs(float(center[0])) < 1e-5 and abs(float(center[1])) < 1e-5, (
        f"正面平行平面不应有横向分量，实际 {center.tolist()}")
    # 整幅图应都是同一个方向（平面是平的）
    assert float(n[..., 2].mean()) < -0.999


class _Cfg:
    def __init__(self, **kw):
        self.lambda_depth_distortion = kw.get("ldd", 0.0)
        self.lambda_normal = kw.get("ln", 0.0)
        self.depth_distortion_mode = kw.get("mode", "tv")


def _fake_rasterization_factory(depth_map, normal_map):
    """伪造 gsplat.rasterization，让 depth/normal 正则可以在 CPU 上被测。

    这也是必要的：真实 gsplat 需要 CUDA，而这两个正则的**数学**是纯张量运算，
    没有理由必须上 GPU 才能验证。

    计数约定：`render_mode` 含 **ED**（`RGB+ED`，训练默认）或 `D`（仅对照）都记到 `depth`；
    **只有当 colors 是逐高斯 3 通道法线**（形如 [N,1,3]）时才算 `normal` ——
    主渲染传的是 SH（[N,1,3] 也是 3 通道，但维度是 [N,K,3]，K 是 SH 系数个数，通常为 16）。
    这里按"N 与传入的 quats 一致且第二维为 1"来区分，避免把主渲染误计。

    ⚠️ 2026-10-25：深度语义改为 **ED（期望深度，与导出统一）**，所以 `RGB+ED` 的返回值是
    **[1,H,W,4]**（RGB + 深度通道），不是旧的 [1,H,W,1]。这个假实现必须跟着契约走 ——
    否则测试会掩盖真实的调用契约变化。
    """
    calls = {"depth": 0, "normal": 0}

    def fake(means=None, quats=None, scales=None, opacities=None, colors=None,
             viewmats=None, Ks=None, width=None, height=None, sh_degree=None,
             render_mode="RGB", **kw):
        H, W = depth_map.shape
        if render_mode in ("D", "RGB+ED", "RGB+D"):
            calls["depth"] += 1
            if render_mode == "RGB+ED":
                out = torch.zeros(1, H, W, 4)
                out[0, ..., 0:3] = 0.5          # RGB 占位
                out[0, ..., 3] = depth_map      # ED 通道
            else:
                out = torch.zeros(1, H, W, 1)
                out[0, ..., 0] = depth_map
            return out, torch.ones(1, H, W, 1), {}
        # 判断这是不是"渲染法线"那一趟：colors 形如 [N,1,3]（第二维为 1）
        is_normal_pass = (colors is not None and colors.dim() == 3
                          and colors.shape[1] == 1)
        if is_normal_pass:
            calls["normal"] += 1
        out = torch.zeros(1, H, W, 3)
        if is_normal_pass:
            out[0] = normal_map * 0.5 + 0.5
        return out, torch.ones(1, H, W, 1), {}

    return fake, calls


def _run_regularizers(monkeypatch_mod, cfg, depth_map, normal_map):
    import gsplat

    fake, calls = _fake_rasterization_factory(depth_map, normal_map)
    orig = gsplat.rasterization
    gsplat.rasterization = fake
    try:
        from gsflow.trainers.gsplat_trainer import two_dgs_regularizers

        H, W = depth_map.shape
        K = torch.tensor([[50.0, 0.0, W / 2.0], [0.0, 50.0, H / 2.0], [0.0, 0.0, 1.0]])
        vm = torch.eye(4)
        res = two_dgs_regularizers(
            cfg=cfg, means=torch.zeros(4, 3), quats=torch.tensor(
                [[1.0, 0, 0, 0]]).repeat(4, 1),
            scales=torch.tensor([[1.0, 1.0, 0.01]]).repeat(4, 1),
            opacities=torch.ones(4), shs=torch.zeros(4, 1, 3),
            kk=K, vm=vm, width=W, height=H,
            alphas=torch.ones(1, H, W, 1), sh_degree=0,
        )
    finally:
        gsplat.rasterization = orig
    return res, calls


def test_depth_distortion_is_a_real_loss() -> None:
    """深度失真项必须随"深度图更不平滑"而**增大**，且不能恒为 0。"""
    import gsplat  # noqa: F401

    H = W = 24
    flat = torch.full((H, W), 2.0)
    ys = torch.arange(H, dtype=torch.float32).view(-1, 1).expand(H, W)
    sloped = 2.0 + 0.02 * ys
    rough = 2.0 + 0.5 * torch.sin(ys * 3.0).expand(H, W)

    nmap = torch.zeros(H, W, 3)
    nmap[..., 2] = -1.0

    r_flat, _ = _run_regularizers(None, _Cfg(ldd=0.05, ln=0.0), flat, nmap)
    r_slope, _ = _run_regularizers(None, _Cfg(ldd=0.05, ln=0.0), sloped, nmap)
    r_rough, _ = _run_regularizers(None, _Cfg(ldd=0.05, ln=0.0), rough, nmap)

    assert r_flat["depth"] is not None
    lf, ls, lr = float(r_flat["depth"]), float(r_slope["depth"]), float(r_rough["depth"])
    assert lf < 1e-6, f"常数深度图的失真应为 0，实际 {lf}"
    assert ls > 0.0, "倾斜深度图应有非零失真"
    assert lr > ls, f"更粗糙的深度图失真应更大：rough={lr} slope={ls}"


def test_depth_distortion_skipped_when_weight_zero() -> None:
    """两个权重都为 0 时不应触发任何额外渲染（也不该返回损失项）。"""
    H = W = 16
    depth = torch.full((H, W), 2.0)
    nmap = torch.zeros(H, W, 3)
    nmap[..., 2] = -1.0
    r, calls = _run_regularizers(None, _Cfg(ldd=0.0, ln=0.0), depth, nmap)
    assert r["depth"] is None and r["normal"] is None
    assert calls["depth"] == 0 and calls["normal"] == 0, "权重为 0 时不应触发额外渲染"

    # 只开深度正则时：只应触发深度渲染，不触发法线渲染
    r2, calls2 = _run_regularizers(None, _Cfg(ldd=0.05, ln=0.0), depth, nmap)
    assert calls2["depth"] == 1 and calls2["normal"] == 0
    assert r2["normal"] is None


def test_normal_consistency_is_not_identically_zero() -> None:
    """法线一致性**不能恒为 0** —— 第一版写成恒等式，等于没加正则。

    直接单测抽出来的 `normal_consistency_loss`：法线一致 -> ~0；
    法线垂直 -> ~1；法线相反 -> ~0（法线正负对"表面朝向"等价，故取绝对值）。
    关键是**不同输入必须给出不同损失**。
    """
    from gsflow.trainers.gsplat_trainer import normal_consistency_loss

    H = W = 8
    w = torch.ones(H, W)
    base = torch.zeros(H, W, 3)
    base[..., 2] = -1.0                      # 朝相机

    same = normal_consistency_loss(base, base, w)
    perp = torch.zeros(H, W, 3)
    perp[..., 0] = 1.0                       # 与 base 垂直
    r_perp = normal_consistency_loss(perp, base, w)
    flip = -base
    r_flip = normal_consistency_loss(flip, base, w)
    # 45 度夹角：损失应为 1 - cos45 ≈ 0.293
    mid = torch.zeros(H, W, 3)
    mid[..., 2] = -0.70710678
    mid[..., 0] = 0.70710678
    r_mid = normal_consistency_loss(mid, base, w)

    assert same is not None and float(same) < 1e-6, f"一致时应为 0，实际 {same}"
    assert abs(float(r_perp) - 1.0) < 1e-5, f"垂直时应为 1，实际 {r_perp}"
    assert float(r_flip) < 1e-6, "反向法线的朝向等价，损失应仍为 0"
    assert abs(float(r_mid) - (1 - 0.70710678)) < 1e-5, f"45 度时应为 0.293，实际 {r_mid}"
    # 核心断言：不能恒为 0
    assert float(r_perp) > float(same) + 0.5

    # 权重为 0 时返回 None（而不是 0，避免"看起来生效了"）
    assert normal_consistency_loss(base, base, torch.zeros(H, W)) is None


def test_depth_distortion_helper_matches_expected_values() -> None:
    """深度失真辅助函数：常数深度为 0；线性斜坡为常数斜率；二阶模式对大跳变更敏感。"""
    from gsflow.trainers.gsplat_trainer import depth_distortion_loss

    H = W = 10
    a = torch.ones(H, W)
    flat = torch.full((H, W), 3.0)
    assert float(depth_distortion_loss(flat, a, "tv")) < 1e-6

    ys = torch.arange(H, dtype=torch.float32).view(-1, 1).expand(H, W)
    slope = 0.1 * ys
    # 一阶 TV = mean(|Δx D|) + mean(|Δy D|)：水平差分为 0，竖直差分为 0.1
    # （两个方向是**相加**而不是平均，所以结果是 0.1）
    v = float(depth_distortion_loss(slope, a, "tv"))
    assert abs(v - 0.1) < 1e-5, f"线性斜坡的 TV 应为 0.1，实际 {v}"
    # 二阶（Laplacian）对线性斜坡为 0
    assert float(depth_distortion_loss(slope, a, "lap")) < 1e-5
    # 二阶对"跳变"敏感：中间一行突变应给出非零
    jump = torch.zeros(H, W)
    jump[H // 2, :] = 1.0
    assert float(depth_distortion_loss(jump, a, "lap")) > 0.0
