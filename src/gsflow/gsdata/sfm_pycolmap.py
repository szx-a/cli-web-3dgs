"""纯 Python 的 SfM 实现（pycolmap 后端），作为外部 COLMAP 失败时的**兜底**。

## 为什么需要（2026-10-24 实测，非权限问题）

本机 `tools/colmap/bin/colmap.exe`（3.12.0 Windows 构建）的 `SiftMatching.*` 选项**绑定失效**：

* 传非法值 → `the argument ('abc') for option '--SiftMatching.max_num_matches' is invalid`
  ⇒ 选项**被识别**；
* 传合法值 `1` / `999999` / 完全不传 → 检查点看到的**永远是 0**：
  `Check failed: max_num_matches > 0 (0 vs. 0)` ⇒ 解析进去的值**没有到达匹配器结构体**；
* 空数据库、无任何 `--SiftMatching.*` 参数也照样崩 ⇒ 与数据、与命令行无关；
* 文件属性 `Archive`、`icacls` 显示 `Authenticated Users:(M)` ⇒ **不是权限问题**。

结论：**该构建的 matcher 子系统坏了**（帮助文本写 32768、运行时是 0）。
修法是换一个 COLMAP 构建，或者**绕开 exe**：pycolmap 是 COLMAP 官方 Python 绑定，
共享同一套 SIFT / 增量建图 / `.bin` 文件格式，产物可被下游直接读取。

实测（img3 406 张 456×290，sequential overlap=12）：**7.3 分钟，406/406 注册，12,705 个 3D 点**，
比外部 COLMAP（`sparse/1`：406 张 / 8,753 点）多 46% 的初始化点。
"""
from __future__ import annotations

import json
from pathlib import Path


def run_pycolmap_sfm(images_dir: Path, work_dir: Path, out_sparse: Path,
                     max_features: int = 8192, max_image_size: int = 2048,
                     camera_model: str = "SIMPLE_RADIAL", overlap: int = 12,
                     use_gpu: bool = False, log=print) -> dict:
    """跑特征→匹配→增量重建，把最好的模型写到 `out_sparse/0`。

    返回元信息 dict（含 `num_registered` / `num_points3D`），供阶段写清单。
    失败时抛 `RuntimeError`，由调用方决定是继续抛还是降级。
    """
    import pycolmap

    images_dir = Path(images_dir)
    work_dir = Path(work_dir)
    out_sparse = Path(out_sparse)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_sparse.mkdir(parents=True, exist_ok=True)
    db = work_dir / "pycolmap.db"
    if db.exists():
        db.unlink()

    log(f"pycolmap {pycolmap.__version__}：特征提取（{images_dir}）")
    reader = pycolmap.ImageReaderOptions()
    try:
        reader.camera_model = camera_model
    except Exception:  # noqa: BLE001
        pass
    extract = pycolmap.FeatureExtractionOptions()
    try:
        extract.sift.max_num_features = int(max_features)
        extract.sift.max_image_size = int(max_image_size)
        extract.use_gpu = bool(use_gpu)
    except Exception:  # noqa: BLE001
        pass
    pycolmap.extract_features(
        database_path=str(db), image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader, extraction_options=extract,
    )

    names = sorted(p.name for p in images_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    consecutive = bool(names) and all(n[:1].isdigit() for n in names)
    match_opts = pycolmap.FeatureMatchingOptions()
    try:
        match_opts.use_gpu = bool(use_gpu)
    except Exception:  # noqa: BLE001
        pass
    if consecutive:
        pair = pycolmap.SequentialPairingOptions()
        try:
            pair.overlap = int(max(2, overlap))
            pair.quadratic_overlap = True
        except Exception:  # noqa: BLE001
            pass
        log(f"pycolmap：序列匹配（overlap={overlap}）")
        pycolmap.match_sequential(database_path=str(db), matching_options=match_opts,
                                  pairing_options=pair)
    else:
        log("pycolmap：穷举匹配（帧编号不连续）")
        pycolmap.match_exhaustive(database_path=str(db), matching_options=match_opts)

    log("pycolmap：增量重建")
    maps = pycolmap.incremental_mapping(
        database_path=str(db), image_path=str(images_dir), output_path=str(out_sparse))
    if not maps:
        raise RuntimeError("pycolmap 没能建出任何模型（图像/匹配质量不足）")
    best = max(maps.values(), key=lambda r: len(r.images))
    best_dir = out_sparse / "0"
    best_dir.mkdir(parents=True, exist_ok=True)
    best.write(str(best_dir))
    meta = {
        "backend": "pycolmap",
        "version": pycolmap.__version__,
        "num_registered": int(len(best.images)),
        "num_points3D": int(len(best.points3D)),
        "sequential": bool(consecutive),
    }
    (out_sparse / "pycolmap_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"pycolmap 完成：注册 {meta['num_registered']} 张、3D 点 {meta['num_points3D']}")
    return meta
