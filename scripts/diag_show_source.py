"""打印 `backproject_depth_to_points` **实际加载的源码**（不是我去读文件）。

动机：手动复现循环体完全正常（valid=337617 -> stride 后 84509），
但调用同名函数却返回 `{'views_used': 0}`。这说明**模块里实际生效的代码**
与磁盘上我以为的那份不一致（例如有第二处定义覆盖、或缓存了旧字节码）。
用 `inspect.getsource` 看真实生效的源码是最直接的判据。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow.gsdata import mesh_export  # noqa: E402

fn = mesh_export.backproject_depth_to_points
print("实际函数所在文件:", inspect.getsourcefile(fn))
print("定义起始行号:", inspect.getsourcelines(fn)[1])
print("=" * 70)
print(inspect.getsource(fn))
