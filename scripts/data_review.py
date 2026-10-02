"""Build a local, paginated image review index from prepared manifests."""

import html
import json
from pathlib import Path
from urllib.parse import quote


def build(root):
    output = root / "review"
    output.mkdir(exist_ok=True)
    links = []
    for name in ("identity", "landmarks", "review", "conflicts", "singletons"):
        rows = [json.loads(line) for line in (root / f"prepared.{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        if name == "identity":
            rows.sort(key=lambda row: (row["cat_id"], row["path"]))
        pages = []
        for start in range(0, len(rows), 100):
            page = f"{name}-{start // 100 + 1:03}.html"
            pages.append(page)
            cards = []
            for row in rows[start:start + 100]:
                label = html.escape(row.get("cat_id") or row["source"])
                path = html.escape(row["path"])
                url = quote("../" + row["path"], safe="/:")
                flags = html.escape(", ".join(row.get("preparation_flags", [])) or row.get("exclusion_reason", ""))
                cards.append(f'<figure><a href="{url}"><img loading="lazy" src="{url}"></a>'
                             f'<figcaption><b>{label}</b><br>{flags}<br><small>{path}</small></figcaption></figure>')
            previous = f'<a href="{name}-{start // 100:03}.html">上一页</a>' if start else ""
            following = f'<a href="{name}-{start // 100 + 2:03}.html">下一页</a>' if start + 100 < len(rows) else ""
            content = f'<nav><a href="index.html">索引</a> {previous} {following}</nav><h1>{name}: {start + 1}–{min(start + 100, len(rows))}</h1><main>{"".join(cards)}</main>'
            write_page(output / page, content)
        links.append(f'<li>{name}: {len(rows)} 张 — ' + " ".join(
            f'<a href="{page}">{i + 1}</a>' for i, page in enumerate(pages)) + '</li>')
    pairs = [json.loads(line) for line in (root / "prepared.near-pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    pair_links = []
    for start in range(0, len(pairs), 50):
        page = f"pairs-{start // 50 + 1:03}.html"
        pair_links.append(f'<a href="{page}">{start // 50 + 1}</a>')
        cards = []
        for pair in pairs[start:start + 50]:
            pictures = ''.join(f'<a href="{quote("../" + pair[field], safe="/:")}"><img loading="lazy" src="{quote("../" + pair[field], safe="/:")}"></a>' for field in ("path_a", "path_b"))
            labels = '<br>'.join(html.escape(pair[field]) for field in ("path_a", "path_b"))
            cards.append(f'<figure>{pictures}<figcaption>候选距离 {pair["distance"]}<br><small>{labels}</small></figcaption></figure>')
        write_page(output / page, '<nav><a href="index.html">索引</a></nav><h1>近重复候选对（不等同于同猫）</h1><main>' + ''.join(cards) + '</main>')
    links.append(f'<li>近重复候选：{len(pairs)} 对 — ' + ' '.join(pair_links) + '</li>')
    write_page(output / "index.html", '<h1>猫识别数据复核</h1><p>按来源和身份查看原图。identity 仅表示通过机械检查；未完成逐图身份审核。近重复候选不能自动认定为同猫。原图保持不变。</p><ul>' + "".join(links) + '</ul>')
    print(output / "index.html")


def write_page(path, body):
    path.write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>数据复核</title>'
                    '<style>body{font:16px sans-serif;margin:24px}main{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}'
                    'figure{margin:0;padding:8px;background:#f3f3f3}img{width:100%;height:220px;object-fit:contain}'
                    'figcaption{overflow-wrap:anywhere}nav{position:sticky;top:0;background:white;padding:12px}a{margin-right:8px}</style>'
                    + body + '</html>', encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, nargs="?", default=Path("data"))
    build(parser.parse_args().data_root)
