"""回归测试：训练策略"配置 vs 实际行为"的校验器。

## 为什么这个测试重要

G109 那个 bug（孤儿 `elif` 让 grad2d 每步被 3D 梯度覆盖）最危险的地方在于：
**没有任何东西失败**。PSNR 照常输出、验收照常通过、日志里只有一条重复 30000 次的
warning。我因此连续几轮把 "img3-crop 的 28.78 dB" 当成 grad2d 的效果，
其实那是 grad3d 跑出来的。

所以这个校验器必须有**负向测试**：伪造一个"打出了退回告警"的作业，
它必须报错；伪造一个正常的作业，它必须通过。否则它只是个恒过的装饰。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _fake_job(base: Path, name: str, *, warn_lines: int,
              counts: list[int], criterion: str = "grad2d",
              iterations: int | None = None) -> Path:
    jd = base / name
    (jd / "logs").mkdir(parents=True)
    (jd / "04_train").mkdir(parents=True)
    msgs = []
    for i, c in enumerate(counts):
        msgs.append({"level": "info",
                     "message": f"step {500 + i * 100} 密度：阈值=2.00e-04 "
                                f"克隆 {max(1, c // 10)} 分裂 {max(1, c // 30)} "
                                f"裁剪 3 -> {c} 个高斯"})
    msgs.extend({"level": "warn", "message": "未取到 means2d 梯度，密度控制退回 3D 梯度"}
                for _ in range(warn_lines))
    (jd / "logs" / "events.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs) + "\n",
        encoding="utf-8")
    job = {"job_id": name,
           "settings_snapshot": {"train": {"densify_criterion": criterion}}}
    (jd / "job.json").write_text(json.dumps(job), encoding="utf-8")
    m = {"psnr": 28.0, "gaussians": counts[-1]}
    # ⚠️ 必须显式给 `iterations`：检查器只在训练足够长（>=8000 迭代）时
    # 才判"致密化没生效"，否则几十个 600~3000 迭代的诊断作业会被误判
    # （这是我第一版的问题，见 docs/03 G113）。
    if iterations is not None:
        m["iterations"] = iterations
    (jd / "04_train" / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    return jd


def _run(job_name: str) -> tuple[int, str]:
    r = subprocess.run([PY, str(ROOT / "scripts" / "check_train_used_config.py"),
                        "--job", job_name],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def test_clean_job_passes() -> None:
    """真实作业 `img3-final`（grad2d 确实生效）必须通过。"""
    if not (ROOT / "workspace" / "img3-final" / "logs").exists():
        import pytest
        pytest.skip("没有 img3-final 作业")
    code, out = _run("img3-final")
    assert "实际判据 : grad2d" in out, out[-800:]
    assert "退回告警 : 0 次" in out, out[-800:]
    assert code == 0, out[-800:]


def test_symptom_job_fails() -> None:
    """真实的历史作业 `img3-crop`（打了 30000 次退回告警）必须被判失败。"""
    if not (ROOT / "workspace" / "img3-crop" / "logs").exists():
        import pytest
        pytest.skip("没有 img3-crop 作业")
    code, out = _run("img3-crop")
    assert code != 0, f"历史作业竟然通过了：\n{out[-800:]}"
    assert "已退回" in out, out[-800:]


def test_synthetic_job_with_warnings_is_flagged(workdir: Path) -> None:
    """负向测试（自造数据）：有退回告警 ⇒ 必须报 FAIL。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_train_used_config as C

    # 直接调 analyze()，但把 ROOT 指到临时目录
    old = C.ROOT
    try:
        C.ROOT = workdir
        base = workdir / "workspace"
        base.mkdir(parents=True)
        _fake_job(base, "sick", warn_lines=50, counts=[9000, 12000, 14000],
                  iterations=30000)
        r = C.analyze("sick")
        assert r["fallback_warn"] == 50
        assert "已退回" in r["effective"]
        assert not r["ok"], r["problems"]

        _fake_job(base, "healthy", warn_lines=0, counts=[9000, 40000, 95000],
                  iterations=30000)
        r2 = C.analyze("healthy")
        assert r2["fallback_warn"] == 0
        assert r2["effective"] == "grad2d"
        assert r2["ok"], r2["problems"]

        # 负向：训练足够长但几乎不加密 ⇒ 必须被抓到
        _fake_job(base, "stunted", warn_lines=0, counts=[9000, 9000, 9100],
                  iterations=30000)
        r3 = C.analyze("stunted")
        assert not r3["ok"], r3["problems"]
        assert any("致密化几乎没生效" in p for p in r3["problems"]), r3["problems"]

        # 反向：训练很短（<8000 迭代）时**不判**致密化不足（避免误报诊断作业）
        _fake_job(base, "short", warn_lines=0, counts=[9000, 9000, 9100],
                  iterations=2000)
        r4 = C.analyze("short")
        assert r4["ok"], r4["problems"]
    finally:
        C.ROOT = old


def test_requested_criterion_reads_from_job_json(workdir: Path) -> None:
    """`job.json` 的 `settings_snapshot.train.densify_criterion` 必须读得到。

    我第一版找的是不存在的文件名（settings.json 等），永远返回 None，
    于是"配置请求"这一半信息是空的，检查器看起来在跑其实没查。
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_train_used_config as C

    jd = workdir / "j"
    jd.mkdir()
    (jd / "job.json").write_text(json.dumps(
        {"settings_snapshot": {"train": {"densify_criterion": "grad3d"}}}),
        encoding="utf-8")
    assert C.requested_criterion(jd) == "grad3d"

    # 拍平形式（spec.params）也要支持
    jd2 = workdir / "j2"
    jd2.mkdir()
    (jd2 / "job.json").write_text(json.dumps(
        {"spec": {"params": {"train.densify_criterion": "grad2d"}}}), encoding="utf-8")
    assert C.requested_criterion(jd2) == "grad2d"

    jd3 = workdir / "j3"
    jd3.mkdir()
    assert C.requested_criterion(jd3) is None
