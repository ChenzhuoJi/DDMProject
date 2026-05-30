"""
生成 MDLM 去噪过程动画 (HTML)
================================
从快照文件读取数据，嵌入到 HTML 模板中，输出可直接在浏览器打开的动画页面。

用法：
    python generate_animation.py

输出：
    figures/denoising_animation.html (可打开)
"""

import re
import json
import os

SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), '..', '..',
                             'snapshots', 'snapshot_seed42.txt')
HTML_TEMPLATE = os.path.join(os.path.dirname(__file__), 'denoising_animation.html')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'denoising_animation.html')


def parse_snapshot(filepath: str):
    """解析快照，返回 [(t, [token_or_None, ...]), ...]"""
    rows = []
    with open(filepath) as f:
        for line in f:
            m = re.match(r't=([0-9.]+)', line)
            if not m:
                continue
            t = float(m.group(1))
            text = line[line.index('):') + 2:].strip()
            segments = re.split(r'(\[MASK\])', text)
            tokens = []
            for seg in segments:
                if seg == '[MASK]':
                    tokens.append(None)
                elif seg.strip():
                    tokens.extend(seg.split())
            rows.append((t, tokens))
    rows.sort(key=lambda x: -x[0])  # t 降序
    return rows


def build_snapshot_json(rows):
    """转为前端 JSON 格式"""
    data = []
    for t, toks in rows:
        data.append({
            't': round(t, 3),
            'pct': round((1 - t) * 100, 1),
            'tokens': [tok if tok is not None else None for tok in toks],
        })
    return data


def inject_into_html(html_path: str, snapshot_json: list, output_path: str):
    """将数据嵌入 HTML 的 JS 变量中"""
    with open(html_path, 'r') as f:
        html = f.read()

    json_str = json.dumps(snapshot_json, ensure_ascii=False)
    # 替换占位符
    placeholder = 'const SNAPSHOT_DATA = null;'
    replacement = f'const SNAPSHOT_DATA = {json_str};'
    if placeholder in html:
        html = html.replace(placeholder, replacement)
    else:
        raise ValueError(f'未找到占位符: {placeholder}')

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'已生成: {output_path}')


def main():
    rows = parse_snapshot(SNAPSHOT_FILE)
    print(f'解析到 {len(rows)} 个时间步')
    print(f'每步 {max(len(t) for _, t in rows)} 个 token')

    snapshot_json = build_snapshot_json(rows)
    inject_into_html(HTML_TEMPLATE, snapshot_json, OUTPUT_FILE)

    file_size = os.path.getsize(OUTPUT_FILE)
    print(f'文件大小: {file_size // 1024} KB')


if __name__ == '__main__':
    main()
