"""从代码里的 DEFAULTS 生成训练参数现状表（markdown），供 docs 与汇报使用。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"F:\3dgs-and-ai")
sys.path.insert(0, str(ROOT / "src"))

from gsflow.config import DEFAULTS  # noqa: E402
from gsflow import presets  # noqa: E402

train = DEFAULTS["train"]

# 说明文字（与代码里的注释一致，避免读代码的人再猜一遍）
DESC = {
    "iterations": "总迭代数",
    "resolution": "训练分辨率（长边）",
    "sh_degree": "球谐阶数（颜色视角相关）",
    "rasterize_mode": "光栅化模式（classic / antialiased）",
    "ssim_weight": "SSIM 权重",
    "lambda_lpips": "LPIPS 感知损失权重（需每步算）",
    "lpips_every": "LPIPS 计算间隔（必须 1）",
    "lpips_size": "LPIPS 输入尺寸",
    "sh_rest_reg": "高阶 SH L2 正则（治斜视彩虹）",
    "opacity_crisp_reg": "不透明度对称正则（推实）",
    "opacity_crisp_warmup": "推实正则起始比例",
    "opacity_floor": "不透明度单边下限",
    "opacity_floor_reg": "单边下限权重",
    "opacity_floor_warmup": "下限起点比例",
    "opacity_init": "不透明度初值",
    "lambda_depth_var": "深度方差正则权重（治朦胧）",
    "depth_var_mode": "深度方差模式",
    "depth_var_warmup": "深度方差起点比例",
    "lambda_grad": "细节梯度损失权重（质感）",
    "grad_warmup": "梯度损失起点比例",
    "lambda_normal": "法线一致性权重（2DGS 面元）",
    "lambda_depth_distortion": "深度失真权重",
    "depth_mode": "深度语义（expected=ED / accum）",
    "depth_distortion_mode": "深度失真实现（tv 代理）",
    "bg_alpha_reg": "掩码外 alpha 抑制",
    "mask_loss_mode": "掩码损失模式（composite / inside）",
    "mask_loss_until": "掩码损失持续比例",
    "mask_erode_px": "损失用掩码腐蚀像素",
    "densify_from": "致密化起点",
    "densify_until": "致密化终点",
    "densify_interval": "致密化间隔",
    "densify_criterion": "致密化判据",
    "densify_grad2d_threshold": "屏幕空间梯度阈值",
    "densify_strategy": "密度控制实现（builtin / official）",
    "cap_max": "高斯数量上限",
    "prune_opacity": "低不透明度剪枝阈值",
    "prune_scale3d": "大尺度剪枝阈值（×scene_scale）",
    "max_scale_ratio": "单步尺度上限（×scene_scale）",
    "max_aspect": "长宽比上限",
    "opacity_reset_interval": "不透明度重置间隔",
    "revised_opacity": "分裂/重置系数",
    "reset_opacity": "重置目标值（0=官方 eps 行为）",
    "auto_extend_densify_when_reset": "重置后顺延致密化窗口",
    "lr_means_scene_scale": "means 学习率乘 scene_scale",
    "lr_schedule": "学习率调度（cosine）",
    "lr_min_ratio": "余弦下限比例",
    "eval_metrics": "是否算指标",
    "eval_interval": "指标间隔",
    "eval_views": "评估视角数",
    "absgrad": "AbsGS 绝对梯度",
    "white_background": "白底合成",
    "backend": "训练后端",
}

rows = []
for k, v in train.items():
    if k in ("data_dir", "out_dir"):
        continue
    rows.append((k, v, DESC.get(k, "")))

print("| 参数 | 默认 | 说明 |")
print("|---|---|---|")

def fmt(v):
    if isinstance(v, bool):
        return "`true`" if v else "`false`"
    if isinstance(v, float):
        s = f"{v:g}"
        return f"`{s}`"
    if isinstance(v, str):
        return f"`{v}`" if v else "`\"\"`"
    return f"`{v}`"

for k, v, d in rows:
    print(f"| `{k}` | {fmt(v)} | {d} |")

print(f"\n（共 {len(rows)} 个 train 参数）\n")
print("| 预设 | 覆盖的关键项 |")
print("|---|---|")
for name in presets.names():
    body = presets.PRESETS.get(name, {})
    if not body:
        print(f"| `{name}` | 与默认一致 |")
    else:
        keys = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in body.items())
        print(f"| `{name}` | {keys} |")
