"""打印指定文件的若干行（转成 ASCII 以免控制台乱码），用于排查。

用法: python scripts/show_lines.py <file> <start> <end>
"""
import sys

path, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, encoding="utf-8", errors="replace") as f:
    lines = f.read().splitlines()
for i in range(max(0, a - 1), min(len(lines), b)):
    txt = lines[i]
    ascii_txt = txt.encode("ascii", "replace").decode("ascii")
    print(f"{i + 1:5d}| {ascii_txt[:130]}")
