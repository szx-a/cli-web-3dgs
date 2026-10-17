"""给 00 追加「§六 执行结果对照」（幂等：已存在则不重复追加）。"""
from __future__ import annotations

import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "00-环境核查与实施规划.md"
MARKER = "## 六、执行结果对照（2026-10 实测更新）"

BODY = """## 六、执行结果对照（2026-10 实测更新）

> 这一节把"计划"与"实际"并排放，避免有人按 §四 的里程碑去操作。
> 每条都指向当前文档位置或 `03-踩坑记录` 的小节号。

### 6.1 里程碑状态

| 计划里程碑 | 状态 | 实际结果 / 与计划的差异 |
|---|---|---|
| M0 环境就绪 | ✅ | venv 是 **Python 3.13.11**（计划 3.11）；torch 2.11+cu128、gsplat 1.5.3 在 sm_120 上编译通过；COLMAP/FFmpeg 已 vendored；`gsflow doctor` 有阻断/注意/提示三级待办 |
| M1 骨架 + 配置/凭据 | ✅ | 配置文件是 `config/config.toml`（**不是** `pipeline.toml`）；默认值唯一来源 `config.DEFAULTS`；事件落 `logs/events.jsonl`，续跑与单阶段重跑可用 |
| M2 图像清理 | ✅ | 五个分割后端可选（classical / rembg / fastsam / sam2 / sam3）；**额外发现**：转台素材必须保留背景（`mask.mode=keep`）否则 COLMAP 特征不足（03/G112）|
| M3 COLMAP | ✅ | 注册率实测 406/406；自动裁剪必须放在 **COLMAP 之前**的帧阶段（03/G112）|
| M4 训练 | ✅ | grad2d 判据（曾被孤儿 `elif` 覆盖，03/G109）；`densify_until=4000`；8GB 卡上 30k 迭代约 10 分钟 |
| M5 导出与查看 | ✅ | 不止 `glb/spz`：还产出 **`.splat`**、**59 维 `model_gaussians.ply`**、离线 `viewer_splat.html`、服务端渲染 `preview_*.png`。查看器**改用社区参考实现**（计划里写的是自研 three.js 方案，后判定为错误决策，03/G116）|
| M6 CLI + Web 同构 | ✅ | GUI 与 CLI 共用同一 Orchestrator；模型页有 **5 个模式**（见 02 §2.1）；新增 `/api/selftest`、`/api/render`、`/api/scene_info` |
| M7 度量/报告/归档 | ✅ | 报告 + `03-踩坑记录`（**123 节**）+ 记忆体（实际挂载名 **`cvcg`**）。度量口径在 2026-10 **改了一次**，见 6.3 |

### 6.2 计划里被现实推翻的判断

| 计划中的判断 | 实际结论 | 依据 |
|---|---|---|
| "自研 three.js 查看器就够" | ❌ 自研 WebGL1 泼溅器连踩 `transpose()`、`int` 索引两个 bug，且**会被误读成模型质量差**；改用 vendored `gaussian-splats-3d` | 03/G110、G114、G116、G121 |
| "按分位数裁剪能提升网格质量" | ❌ 分位数会先裁掉**稀疏但真实**的屋檐/翼角：网格只剩 0.85% 体积、表面积 1.72；改 1.0 后 18.10 | 03/G119 |
| "背景无监督是网格破碎的主因之一"（于是加 composite）| 部分对，但**实现有顺序 bug**，composite 对 L1 从未生效 | 03/G123 |
| "`mask_loss_mode=inside` 能让背景不参与训练" | ❌ 负结果：亮背景占比 61.2% → 59.6%，没有解决目标问题 | 03/G122 |
| "grad3d 判据更稳" | ❌ 与 grad2d 质量无显著差异，但多 48% 高斯 | 03/G117 |
| "降低 densify 阈值能提质量" | ❌ 阈值 1e-4 多 89% 高斯、质量无提升 | 03/G117 |
| "PSNR 28.58 是这个模型的画质" | ❌ 那是掩码无效时的**全图**数字（63% 画面是均匀亮背景）；掩码内才是 20.47 | 03/G123、G124 |

### 6.3 当前质量口径（**引用旧数字前必读**）

| 指标 | 掩码无效时期（历史） | **掩码生效（现在）** |
|---|---|---|
| 全图 PSNR | 28.581 dB | 3.412 dB（**无意义**）|
| **掩码内 PSNR** | 当时没有这个指标 | **20.472 dB** |
| 全图 SSIM | 0.9521 | 0.3673（**无意义**）|
| **掩码内 SSIM** | — | **0.9462** |
| 高斯数 | 94,598 | 145,723 |
| 亮高斯占视觉权重 | 62.8% | **11.4%** |

结论：**质量判据用 `psnr_masked` / `ssim_masked`**；验收脚本用 18 dB 作回归守门
（标注"阈值待标定"），主判据是掩码内 SSIM ≥ 0.93。

### 6.4 §五 那 5 个"需要确认"的事项，最终怎么定的

| # | 计划里的问题 | 实际决定 |
|---|---|---|
| 1 | CUDA Toolkit 是否装 | **装了**（便携版进 `tools/cuda`，免管理员），gsplat 内核编译通过 |
| 2 | Python 版本 | **3.13.11**（不是 3.11），torch 2.11+cu128 可用 |
| 3 | 分割后端优先级 | 先 rembg/FastSAM 打通，SAM2/SAM3 接口保留；**实际主用 fastsam** |
| 4 | 数据/模型落盘 | `F:\\3dgs-and-ai` 为唯一写入根，`D:\\3dgs-workspace` 是同名 junction；`TEMP`/`TMP` 必须指回 `env\\tmp` |
| 5 | GUI 栈 | **FastAPI + 原生 HTML/JS**（无构建步骤、无外网 CDN），与计划一致；高斯渲染用 vendored 库 |

### 6.5 仍然开着的项（下一阶段）

1. **容量集中度**：贡献 80% 视觉权重的高斯只占 **1.6%**（2,396 / 145,723），
   最大尺度 3.19 是中位（0.0134）的 238 倍 ⇒ 候选方案：尺度正则 / 按尺度裁剪 /
   不透明度-面积联合剪枝。
2. **文档概念级守门**：`check_docs_consistency.py` 目前校验参数默认值，
   还应校验"关键概念是否被文档覆盖"（本轮已加 `scripts/check_docs_concepts.py`）。
3. **掩码内 PSNR 的社区基准**：现有 18 dB 阈值是回归守门，不是达标线。
"""


def main() -> int:
    t = DOC.read_text(encoding="utf-8")
    if MARKER in t:
        print("已存在 §六，跳过")
        return 0
    if not t.endswith("\n"):
        t += "\n"
    t += "\n---\n\n" + BODY
    DOC.write_text(t, encoding="utf-8")
    print(f"已追加 §六（{len(BODY.splitlines())} 行），当前 {len(t.splitlines())} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
