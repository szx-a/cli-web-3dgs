"""训练后端：原版 3DGS（graphdeco-inria/gaussian-splatting）以及任意外部训练器。

用途：
  * 需要严格复现论文结果时；
  * 用户手上已经有某个加速训练仓库（LiteGS / faster-gaussian-splatting 等）时，
    用 external 模式直接调它的 train.py。

现实提醒（RTX 50 / sm_120）：
  原版的光栅化扩展在 Blackwell 上需要显式指定 sm_120 重新编译，
  且官方仓库近年未更新，编译失败是常态。默认请用 gsplat 后端。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from ..errors import StageError
from ..pipeline.context import StageContext
from ..tools.locate import run_tool


class OfficialBackend:
    name = "official"

    def _repo(self, ctx: StageContext) -> Path:
        repo = str(ctx.cfg("official.repo", "") or "")
        if repo:
            return Path(repo)
        candidate = Path(str(ctx.cfg("tools.repo_root", "") or "")) / "gaussian-splatting"
        return candidate

    def available(self, ctx: StageContext) -> tuple[bool, str]:
        repo = self._repo(ctx)
        if not (repo / "train.py").exists():
            return False, (
                f"未找到原版 3DGS 仓库（期望 {repo}\\train.py）。"
                "git clone https://github.com/graphdeco-inria/gaussian-splatting 后，"
                "在 config 里设置 tools.repo_root 指向其父目录；"
                "并注意 sm_120 需要重新编译子模块："
                "pip install submodules/diff-gaussian-rasterization submodules/simple-knn "
                "（编译前设置 TORCH_CUDA_ARCH_LIST=12.0）"
            )
        return True, ""

    def run(self, ctx: StageContext, data_dir: Path, out_dir: Path, iterations: int) -> dict:
        ok, why = self.available(ctx)
        if not ok:
            raise StageError("train", f"原版后端不可用：{why}", "或者改用 train.backend=gsplat")
        repo = self._repo(ctx)
        cmd = [
            sys.executable, str(repo / "train.py"),
            "-s", str(data_dir),
            "-m", str(out_dir),
            "--iterations", str(iterations),
            "--eval" if bool(ctx.cfg("eval", True)) else "--no-eval",
            "--resolution", str(int(ctx.cfg("resolution", 1600) or 0)),
        ]
        if bool(ctx.cfg("white_background", False)):
            cmd.append("--white_background")
        log = ctx.logs_dir() / "train_official.log"

        def on_line(line: str) -> None:
            if any(k in line for k in ("Training", "Evaluating", "PSNR", "ITER")):
                ctx.emit(line, level="debug")

        ctx.info("调用原版训练器：" + " ".join(cmd[:6]) + " ...")
        code = run_tool(cmd, cwd=repo, log_file=log, on_line=on_line, env_extra={"TORCH_CUDA_ARCH_LIST": "12.0"})
        if code != 0:
            raise StageError("train", f"原版训练器失败（exit={code}）", f"查看日志 {log}")
        ply = out_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        return {"ply": str(ply), "backend": "official", "iterations": iterations}


class ExternalBackend:
    """任意外部训练脚本：只要它接受 -s/--source_path 与 -m/--model_path。"""

    name = "external"

    def available(self, ctx: StageContext) -> tuple[bool, str]:
        entry = str(ctx.cfg("external.entry", "") or ctx.params.get("train.external_entry", "") or "")
        if not entry or not Path(entry).exists():
            return False, "未配置外部训练器入口脚本（train.external_entry 指向 train.py 的绝对路径）"
        return True, ""

    def run(self, ctx: StageContext, data_dir: Path, out_dir: Path, iterations: int) -> dict:
        ok, why = self.available(ctx)
        if not ok:
            raise StageError("train", f"外部训练器不可用：{why}", "在 GUI 设置页或 CLI --set train.external_entry=... 填写")
        entry = str(ctx.cfg("external.entry", "") or ctx.params.get("train.external_entry", ""))
        extra = str(ctx.cfg("extra_args", "") or "")
        cmd = [sys.executable, entry, "-s", str(data_dir), "-m", str(out_dir), "--iterations", str(iterations)]
        if extra:
            cmd += extra.split()
        log = ctx.logs_dir() / "train_external.log"
        code = run_tool(cmd, cwd=str(Path(entry).parent), log_file=log,
                        on_line=lambda l: ctx.emit(l, level="debug"),
                        env_extra={"TORCH_CUDA_ARCH_LIST": "12.0"})
        if code != 0:
            raise StageError("train", f"外部训练器失败（exit={code}）", f"查看日志 {log}")
        ply = out_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        return {"ply": str(ply), "backend": "external", "iterations": iterations}
