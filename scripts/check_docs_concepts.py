"""检查文档的**概念覆盖**：新机制/新口径必须出现在对应文档里。

## 为什么需要它（这一条是被真实事故逼出来的）

`check_docs_consistency.py` 原本只校验**参数默认值**（README 表格 vs `config.DEFAULTS`），
于是出现了这种局面：

    参数默认值全部一致 ✅        但文档里 0 处提到 psnr_masked、refviewer、auto_crop

也就是说：**数值同步了，概念没同步**。结果是我在聊天里反复解释"质量要看掩码内指标"，
而 `01/02/05` 三篇文档还写着"全图 PSNR 27~31 dB 才达标"，任何人照着文档做都会得出错误结论。

## 判据

每个概念声明"必须出现在哪些文档里"。检查的是**字面出现**
（`psnr_masked`、`refviewer` 这类标识符足够独特，不需要模糊匹配）。

**必须带负向自测**（本项目硬规矩）：`--selftest` 会造一份"故意缺概念"的假文档，
断言检查器**会失败**。否则这又是一个"永远通过"的假检查。

用法:
  python scripts/check_docs_concepts.py            # 正常检查
  python scripts/check_docs_concepts.py --selftest # 负向自测
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# (概念, 说明, 必须出现在哪些文档)
REQUIRED: list[tuple[str, str, tuple[str, ...]]] = [
    # ---- 质量口径（2026-10 变了，最容易漏）----
    ("psnr_masked", "掩码内 PSNR（掩码生效后唯一有意义的画质指标）",
     ("01-架构说明.md", "02-操作手册.md", "05-训练框架.md", "00-环境核查与实施规划.md",
      "../README.md")),
    ("ssim_masked", "掩码内 SSIM",
     ("02-操作手册.md", "05-训练框架.md")),
    ("mask_loss_mode", "掩码损失的两种口径",
     ("02-操作手册.md", "05-训练框架.md", "../README.md")),
    # ---- 渲染路径 ----
    ("refviewer", "高斯显示的适配层（参考实现）",
     ("01-架构说明.md", "02-操作手册.md", "04-文件树.md")),
    ("gaussian-splats-3d", "vendored 的社区参考渲染库",
     ("01-架构说明.md", "04-文件树.md")),
    ("vendor", "离线第三方前端库目录",
     ("01-架构说明.md", "02-操作手册.md", "04-文件树.md")),
    # ---- 导出产物 ----
    ("model_gaussians.ply", "完整 59 维高斯 PLY",
     ("01-架构说明.md", "02-操作手册.md")),
    (".splat", "32 字节/高斯的紧凑格式",
     ("01-架构说明.md", "02-操作手册.md")),
    # ---- 自动裁剪必须在 COLMAP 之前 ----
    ("auto_crop", "帧阶段的时间方差自动裁剪",
     ("01-架构说明.md", "02-操作手册.md", "../README.md")),
    # ---- 训练判据 ----
    ("grad2d", "屏幕空间梯度判据（默认）",
     ("05-训练框架.md", "02-操作手册.md")),
    ("densify_grad2d_threshold", "grad2d 的官方阈值 2e-4",
     ("05-训练框架.md",)),
    # ---- 网格裁剪的方向（曾被我自己写反）----
    ("mesh_crop_quantile", "网格聚焦的分位数裁剪（默认 1.0）",
     ("02-操作手册.md", "05-训练框架.md", "../README.md")),
    # ---- 运维 ----
    ("selftest", "前端打不开时的第一站接口",
     ("02-操作手册.md",)),
    ("cvcg", "记忆体挂载名（曾写错成 cgcv）",
     ("00-环境核查与实施规划.md",)),
    # ---- 交接页本身也不能漂移（它是新会话的入口）----
    ("psnr_masked", "交接页必须写明当前画质口径",
     ("06-交接与下一步.md",)),
    ("final_acceptance", "交接页必须给出验收命令",
     ("06-交接与下一步.md",)),
    ("make_tree", "交接页必须提醒文件树是自动生成的",
     ("06-交接与下一步.md",)),
    # ---- 2026-10 针状 / 黑壳修复（docs/03 G125）----
    ("max_scale_ratio", "每步尺度上限（防针状高斯）",
     ("02-操作手册.md", "05-训练框架.md", "../README.md")),
    ("bg_alpha_reg", "掩码外 alpha 惩罚（防黑壳）",
     ("02-操作手册.md", "05-训练框架.md", "../README.md")),
    ("check_free_view", "自由视角/越界雾丝检查",
     ("06-交接与下一步.md",)),
]


def _read(rel: str) -> str | None:
    p = (DOCS / rel).resolve()
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def check() -> list[str]:
    """返回失败描述列表；空列表 = 全部通过。"""
    fails: list[str] = []
    for concept, why, docs in REQUIRED:
        for rel in docs:
            text = _read(rel)
            if text is None:
                fails.append(f"文档缺失：{rel}（概念 {concept}）")
                continue
            if _absent(text, concept):
                fails.append(f"{rel} 缺少概念 `{concept}` —— {why}")
    return fails


def _absent(text: str, concept: str) -> bool:
    """核心判定：概念是否缺失。抽出来便于在**内存字符串**上自测。"""
    return concept not in text


def selftest() -> int:
    """负向自测：同一套判定逻辑必须能在"故意缺失"的文本上报缺失。

    ⚠️ 不写临时文件：本项目沙箱对 `env/tmp` 下新建的临时目录会拒绝写入
    （实测 PermissionError）。自测只需要判定逻辑，用内存字符串更简单也更稳。
    """
    print("=== 负向自测：检查器必须能发现缺失 ===")
    ok = True
    empty = "这里什么都没有，没有任何关键概念。\n"
    for concept in ("psnr_masked", "refviewer", "mesh_crop_quantile"):
        detected = _absent(empty, concept)
        print(f"  [{'OK  ' if detected else 'FAIL'}] 缺 `{concept}` 被检出")
        ok = ok and detected
    # 反向：概念存在时不得误报
    present = "psnr_masked / refviewer / mesh_crop_quantile 都在。\n"
    for concept in ("psnr_masked", "refviewer", "mesh_crop_quantile"):
        no_fp = not _absent(present, concept)
        print(f"  [{'OK  ' if no_fp else 'FAIL'}] 存在 `{concept}` 时不误报")
        ok = ok and no_fp
    print("  负向自测" + ("通过" if ok else " **失败**"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    fails = check()
    total = sum(len(d) for _, _, d in REQUIRED)
    print(f"  检查 {len(REQUIRED)} 个概念 / {total} 处文档引用")
    if args.verbose:
        for concept, why, docs in REQUIRED:
            print(f"    {concept:<24} → {', '.join(docs)}")
    if fails:
        print(f"  **{len(fails)} 处缺失**：")
        for f in fails:
            print(f"    - {f}")
        return 1
    print("  关键概念全部已覆盖")
    return 0


if __name__ == "__main__":
    sys.exit(main())
