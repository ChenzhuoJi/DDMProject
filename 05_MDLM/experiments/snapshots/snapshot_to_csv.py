"""
snapshot_to_csv.py — MDLM 快照 → CSV 解析工具

将快照文件解析为宽表 CSV：每行一个时间步，
列为 time, Position_0, Position_1, …, Position_{N-1}。

用法:
    python snapshot_to_csv.py [文件或通配符...]
    无参数时处理 snapshots/ 下所有 snapshot_*.txt

输出:
    csv/ 目录下同名 .csv
"""

import os
import sys
import re
import csv
from glob import glob

SNAPSHOT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_snapshot(filepath):
    rows = []
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('=='):
            continue
        m = re.match(r't=([\d.]+)\s+\(\s*\d+%\s*完成\):\s*(.*)', line)
        if not m:
            continue
        time = float(m.group(1))
        tokens = m.group(2).strip().split()
        rows.append((time, tokens))
    return rows


def snapshot_to_csv(filepath, output_dir):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(output_dir, f'{basename}.csv')

    rows = parse_snapshot(filepath)
    if not rows:
        print(f'  ⚠ 跳过 {basename}: 无有效数据')
        return

    n_tokens = len(rows[0][1])
    os.makedirs(output_dir, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time'] + [f'Position_{i}' for i in range(n_tokens)])
        for time, tokens in rows:
            writer.writerow([time] + tokens)

    print(f'  ✓ {basename}.csv  ({len(rows)} 行 × {n_tokens} 列)')
    return out_path


def main():
    args = sys.argv[1:]
    if args:
        files = []
        for pat in args:
            files.extend(glob(pat))
    else:
        files = sorted(glob(os.path.join(SNAPSHOT_DIR, 'snapshot_*.txt')))

    if not files:
        print('未找到快照文件。指定文件或置于 snapshots/ 目录下。')
        return

    output_dir = os.path.join(SNAPSHOT_DIR, 'csv')
    print(f'输出目录: {output_dir}/')
    for fp in files:
        snapshot_to_csv(fp, output_dir)


if __name__ == '__main__':
    main()
