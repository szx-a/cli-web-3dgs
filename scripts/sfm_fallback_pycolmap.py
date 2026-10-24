"""纯 Python 的 COLMAP 兜底：用 pycolmap 跑特征→匹配→增量重建。

## 为什么需要（2026-10-24 实测）

本机 `tools/colmap/bin/colmap.exe`（3.12.0）的 **`--SiftMatching.*` 参数完全不生效**：
即使显式传 `--SiftMatching.max_num_matches 32768`（或 `=5000`、`--SiftMatching.use_gpu 0`），
运行时报错仍是

    Check failed: max_num_matches > 0 (0 vs. 0)      # sift.cc:89
    Check failed: options_.Check()                    # sift.cc:1176 / 978

连**空数据库 + 不带任何参数**都崩，说明不是我们的取值问题，而是该二进制的选项绑定坏了。
后果：**任何新数据都无法匹配**，整条流水线在第③步就死 —— 这是鲁棒性上的致命单点。

兜底方案：`pycolmap`（纯 Python 绑定，4.2.1）。它自己做特征/匹配/建图，
产物仍是标准 `sparse/0/{cameras,images,points3D}.bin`，下游阶段无需改动。

用法:
    python scripts/sfm_fallback_pycolmap.py --images <dir> --workspace <dir> --out <sparse_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--workspace", required=True, help="放 database.db 与中间产物")
    ap.add_argument("--out", required=True, help="输出 sparse 目录（会写 sparse/0）")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--max-image-size", type=int, default=2048)
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--masks", default="", help="可选：掩码目录（pycolmap 不支持就不传）")
    ap.add_argument("--use-gpu", action="store_true")
    args = ap.parse_args()

    import pycolmap

    images = Path(args.images)
    ws = Path(args.workspace)
    out = Path(args.out)
    ws.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    db = ws / "pycolmap.db"
    if db.exists():
        db.unlink()

    print(f"  pycolmap {pycolmap.__version__}")
    print(f"  图像：{images}")

    # 1) 特征
    reader = pycolmap.ImageReaderOptions()
    try:
        reader.camera_model = args.camera_model
    except Exception:  # noqa: BLE001
        pass
    extract = pycolmap.FeatureExtractionOptions()
    try:
        extract.sift.max_num_features = int(args.max_features)
        extract.sift.max_image_size = int(args.max_image_size)
        extract.use_gpu = bool(args.use_gpu)
    except Exception:  # noqa: BLE001
        pass
    pycolmap.extract_features(
        database_path=str(db), image_path=str(images),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader, extraction_options=extract,
    )
    print("  特征提取完成")

    # 2) 匹配：连续编号帧用 sequential（快得多），否则穷举
    #
    # 实测：406 张做穷举 = 8.2 万对，在本机要跑很久；
    # 而轨道拍摄的帧序本来就是连续的，只需与前后几帧匹配。
    names = sorted(p.name for p in images.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    consecutive = bool(names) and all(n[:1].isdigit() for n in names)
    match_opts = pycolmap.FeatureMatchingOptions()
    try:
        match_opts.use_gpu = bool(args.use_gpu)
    except Exception:  # noqa: BLE001
        pass
    if consecutive:
        # pycolmap 4.x 的参数名是 pairing_options（SequentialPairingOptions）
        seq_opts = pycolmap.SequentialPairingOptions()
        try:
            seq_opts.overlap = 12
            seq_opts.quadratic_overlap = True
        except Exception:  # noqa: BLE001
            pass
        pycolmap.match_sequential(
            database_path=str(db),
            matching_options=match_opts,
            pairing_options=seq_opts,
        )
        print("  特征匹配完成（sequential, overlap=12）")
    else:
        pycolmap.match_exhaustive(database_path=str(db), matching_options=match_opts)
        print("  特征匹配完成（exhaustive）")

    # 3) 增量重建
    maps = pycolmap.incremental_mapping(
        database_path=str(db), image_path=str(images), output_path=str(out),
    )
    if not maps:
        print("  建图失败：没有可用重建")
        return 1
    # pycolmap 4.x 没有 num_registered_images()；用 len(rec.images)
    best = max(maps.values(), key=lambda r: len(r.images))
    best_dir = out / "0"
    best_dir.mkdir(parents=True, exist_ok=True)
    best.write(str(best_dir))
    meta = {
        "backend": "pycolmap",
        "version": pycolmap.__version__,
        "num_registered": int(len(best.images)),
        "num_points3D": int(len(best.points3D)),
    }
    (out / "pycolmap_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"  重建完成：注册 {meta['num_registered']} 张、3D 点 {meta['num_points3D']}")
    print(f"  输出：{best_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
