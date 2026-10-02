"""命令行入口：与 Web GUI 共用同一套流程核心。

约定：CLI 的每个参数都能在 GUI 里找到对应控件，反之亦然；
两者都只是构造 JobSpec + 点路径覆盖，然后交给 Orchestrator。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import credentials, paths
from .config import load as load_settings
from .config import write_default_config
from .errors import GsflowError
from .events import EventBus, console_sink
from .pipeline import registry
from .stages import register_all  # noqa: F401  触发阶段注册
from .tools import doctor as doctor_mod

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gsflow",
        description="AI 图像清理 + 3DGS 三维重建全流程（流程核心 / CLI / Web 共用同一实现）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  gsflow doctor\n"
            "  gsflow run --video D:\\data\\room.mp4 --stages frames,mask,colmap\n"
            "  gsflow run --images D:\\data\\photos --set colmap.matcher=exhaustive --set train.iterations=15000\n"
            "  gsflow web --port 8787\n"
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- init ----
    sp = sub.add_parser("init", help="生成默认配置与目录结构")
    sp.add_argument("--force", action="store_true", help="覆盖已存在的配置文件")

    # ---- doctor ----
    sp = sub.add_parser("doctor", help="环境自检（软硬件/工具/依赖/待办）")
    sp.add_argument("--json", action="store_true", help="以 JSON 输出（供脚本消费）")
    sp.add_argument("--save", action="store_true", help="同时写入 docs/env-report.json")

    # ---- stages ----
    sp = sub.add_parser("stages", help="列出所有阶段及其输入输出")

    # ---- run ----
    sp = sub.add_parser("run", help="执行一次重建作业")
    src = sp.add_mutually_exclusive_group()
    src.add_argument("--video", help="输入视频文件")
    src.add_argument("--images", help="输入图片目录（或单张图片）")
    sp.add_argument("--job-id", help="指定作业 id（默认时间戳+随机）")
    sp.add_argument("--title", default="3DGS 重建", help="作业标题")
    sp.add_argument("--stages", default="", help="要执行的阶段，逗号分隔；默认全部")
    sp.add_argument("--force", default="", help="强制重跑的阶段，逗号分隔（忽略已有产物）")
    sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="点路径覆盖配置，可重复，如 --set train.iterations=7000")
    sp.add_argument("--dry-run", action="store_true", help="只打印将执行的阶段，不真正跑")
    sp.add_argument("--no-halt", action="store_true", help="某阶段失败后继续执行后续阶段")
    sp.add_argument("--quiet", action="store_true", help="只输出 warn 及以上事件")

    # ---- jobs ----
    sp = sub.add_parser("jobs", help="列出历史作业")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("status", help="查看某个作业状态")
    sp.add_argument("job_id")
    sp.add_argument("--json", action="store_true")

    # ---- creds ----
    sp = sub.add_parser("creds", help="查看/写入凭据（API Key / HF Token / 镜像 / 代理）")
    sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="写入一条凭据")
    sp.add_argument("--show", action="store_true", help="显示脱敏后的当前配置")

    # ---- web ----
    sp = sub.add_parser("web", help="启动 Web GUI（与 CLI 同构）")
    sp.add_argument("--host", default="")
    sp.add_argument("--port", type=int, default=0)
    sp.add_argument("--open", action="store_true", help="启动后自动打开浏览器")

    # ---- export ----
    sp = sub.add_parser("export", help="对已有作业单独跑导出阶段")
    sp.add_argument("job_id")
    sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    return p


def _parse_sets(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式，收到：{item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _split_csv(text: str) -> list[str]:
    return [x.strip() for x in (text or "").split(",") if x.strip()]


# --------------------------------------------------------------------- 命令
def cmd_init(args) -> int:
    paths.ensure_dirs()
    cfg = paths.CONFIG_DIR / "config.toml"
    if cfg.exists() and not args.force:
        print(f"配置已存在：{cfg}（用 --force 覆盖）")
    else:
        write_default_config(cfg)
        print(f"已写入默认配置：{cfg}")
    credentials.save_credentials({})
    print(f"凭据文件：{credentials.CRED_FILE}（当前为空，可用 `gsflow creds --set hf_token=xxx` 或 GUI 填写）")
    print(f"工作区：{paths.WORKSPACE_DIR}")
    return EXIT_OK


def cmd_doctor(args) -> int:
    report = doctor_mod.collect_report(load_settings())
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(doctor_mod.render_text(report))
    if args.save:
        p = doctor_mod.write_report(report)
        print(f"报告已写入：{p}")
    blockers = [x for x in report["plan"] if x["severity"] == "blocker"]
    return EXIT_FAIL if blockers else EXIT_OK


def cmd_stages(args) -> int:
    for info in registry.describe():
        print(f"{info['name']:<8} {info['title']}")
        print(f"         依赖: {', '.join(info['requires']) or '无'}")
        print(f"         说明: {info['description']}")
        print(f"         输入: {', '.join(i['path'] for i in info['inputs']) or '无'}")
        print(f"         输出: {', '.join(o['path'] for o in info['outputs']) or '无'}")
    return EXIT_OK


def cmd_run(args) -> int:
    from .pipeline.orchestrator import JobSpec, Orchestrator

    paths.ensure_dirs()
    sets = _parse_sets(args.set)
    settings = load_settings(sets)
    spec = JobSpec(
        input_path=args.video or args.images or "",
        images_dir=args.images or "",
        job_id=args.job_id or "",
        title=args.title,
        stages=_split_csv(args.stages),
        params=sets,
        force_stages=_split_csv(args.force),
        halt_on_error=not args.no_halt,
    )
    events = EventBus()
    sink = console_sink()
    if args.quiet:
        events.subscribe(lambda e: sink(e) if e.level in ("warn", "error", "success") else None)
    else:
        events.subscribe(sink)

    orch = Orchestrator(spec, settings=settings, events=events, dry_run=args.dry_run)
    try:
        result = orch.run()
    except KeyboardInterrupt:
        print("\n已中断（产物保留，可用相同 --job-id 续跑）")
        return EXIT_FAIL
    except GsflowError as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return EXIT_FAIL
    print(json.dumps({k: v for k, v in result.items() if k != "stages"}, ensure_ascii=False))
    return EXIT_OK if result["status"] == "done" else EXIT_FAIL


def cmd_jobs(args) -> int:
    from .pipeline.orchestrator import list_jobs

    jobs = list_jobs(Path(str(load_settings().get("project.workspace_dir"))))
    if args.json:
        print(json.dumps(jobs, ensure_ascii=False, indent=2))
        return EXIT_OK
    if not jobs:
        print(f"还没有任何作业（工作区：{paths.WORKSPACE_DIR}）")
        return EXIT_OK
    print(f"{'job_id':<26}{'状态':<10}{'标题':<20}阶段 / 失败原因")
    for j in jobs:
        stages = j.get("stages", {})
        brief = " ".join(f"{k}:{v.get('status', '?')}" for k, v in stages.items())
        failure = ""
        for v in stages.values():
            if v.get("status") == "failed":
                failure = str(v.get("message", ""))[:70]
                break
        print(f"{j.get('job_id', ''):<26}{j.get('status', ''):<10}{str(j.get('title', ''))[:18]:<20}{failure or brief}")
    return EXIT_OK


def cmd_status(args) -> int:
    from .pipeline.orchestrator import load_job

    data = load_job(args.job_id, Path(str(load_settings().get("project.workspace_dir"))))
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"作业 {data.get('job_id')} [{data.get('status')}] {data.get('title')}")
    print(f"目录：{data.get('job_dir')}")
    for name, info in (data.get("stages") or {}).items():
        print(f"  {name:<8} {info.get('status', '?'):<10} {info.get('message', '')}")
    return EXIT_OK


def cmd_creds(args) -> int:
    if args.set:
        data = _parse_sets(args.set)
        path = credentials.save_credentials(data)
        print(f"已写入 {len(data)} 项到 {path}")
    masked = credentials.mask_credentials()
    for k, v in masked.items():
        print(f"  {k:<16} {v if v else '（未设置）'}")
    return EXIT_OK


def cmd_web(args) -> int:
    from .web.app import serve

    settings = load_settings()
    host = args.host or str(settings.get("web.host", "127.0.0.1"))
    port = args.port or int(settings.get("web.port", 8787) or 8787)
    serve(host=host, port=port, open_browser=args.open or bool(settings.get("web.open_browser", False)))
    return EXIT_OK


def cmd_export(args) -> int:
    from .pipeline.orchestrator import JobSpec, Orchestrator

    sets = _parse_sets(args.set)
    spec = JobSpec(job_id=args.job_id, title="导出", stages=["export"], params=sets, force_stages=["export"])
    events = EventBus()
    events.subscribe(console_sink())
    orch = Orchestrator(spec, settings=load_settings(sets), events=events)
    result = orch.run()
    return EXIT_OK if result["status"] == "done" else EXIT_FAIL


COMMANDS = {
    "init": cmd_init,
    "doctor": cmd_doctor,
    "stages": cmd_stages,
    "run": cmd_run,
    "jobs": cmd_jobs,
    "status": cmd_status,
    "creds": cmd_creds,
    "web": cmd_web,
    "export": cmd_export,
}


def main(argv: list[str] | None = None) -> int:
    # 必须在导入任何会读环境变量的第三方库之前生效
    # （ultralytics 的 YOLO_CONFIG_DIR、matplotlib 的 MPLCONFIGDIR 等）
    paths.apply_runtime_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS[args.cmd]
    try:
        return handler(args)
    except GsflowError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
