"""LPIPS 实现的「已知答案」自检 —— 不依赖训练、不依赖真实数据。

为什么必须这样测：感知损失最容易出的问题是**静默无效**（权重缺失→退化成常数、
归一化写错→数值恒定、不可反传→梯度为 0）。这些都不会报错，只会让训练"看起来开了正则
但毫无作用"。所以这里用**可以推理出答案**的输入：

  1. 完全相同的图  -> 距离必须≈0；
  2. 差异越大      -> 距离必须单调更大；
  3. 可反传        -> 对 pred 的梯度必须非零；
  4. 缺失权重时    -> 构造必须抛异常（由调用方退化为"不使用"，而不是给 0）。

用法: scripts\\gpu_python.cmd scripts\\check_lpips.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from gsflow.paths import apply_runtime_env
    apply_runtime_env()
    import torch

    from gsflow.trainers.perceptual import vgg_weights_present

    bad = 0
    print(f"VGG16 权重就绪 = {vgg_weights_present()}")
    if not vgg_weights_present():
        print("  **缺权重**：先跑 python scripts\\fetch_torch_weights.py")
        return 1

    from gsflow.trainers.perceptual import get_lpips

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = get_lpips(dev)
    print(f"设备 {dev}")

    H = W = 128
    torch.manual_seed(0)
    a = torch.rand(1, 3, H, W, device=dev)
    b = a.clone()
    c = (a * 0.9).clamp(0, 1)                      # 轻微不同
    d = torch.rand(1, 3, H, W, device=dev)         # 完全不同

    with torch.no_grad():
        v_same = float(lp(b, a))
        v_near = float(lp(c, a))
        v_far = float(lp(d, a))
    print(f"  相同图      LPIPS = {v_same:.6e}   （应≈0）")
    print(f"  轻微不同    LPIPS = {v_near:.6e}")
    print(f"  完全不同    LPIPS = {v_far:.6e}")
    if v_same > 1e-5:
        print("  **FAIL** 相同图距离不为 0"); bad += 1
    if not (v_near < v_far):
        print("  **FAIL** 距离未随差异单调增长"); bad += 1
    if v_far <= 0:
        print("  **FAIL** 距离恒为 0（归一化或层选取写错？）"); bad += 1

    # 可反传
    pred = a.clone().requires_grad_(True)
    loss = lp(pred, d)
    loss.backward()
    g = float(pred.grad.abs().mean()) if pred.grad is not None else 0.0
    print(f"  梯度均值 = {g:.6e}   （必须 > 0，否则等于开了但学不动）")
    if not (g > 0):
        print("  **FAIL** 不可反传或梯度为 0"); bad += 1

    # 端到端一致性：同一对图两次调用结果应一致（无随机性）
    with torch.no_grad():
        v2 = float(lp(c, a))
    if abs(v2 - v_near) > 1e-9:
        print(f"  **FAIL** 结果不可复现：{v_near} vs {v2}"); bad += 1
    else:
        print("  可复现性 OK")

    print(f"\n结论: {'LPIPS 实现通过全部自检' if bad == 0 else f'**{bad} 项失败**'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
