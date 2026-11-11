"""把 notes/ 中按两位课号命名的 Markdown 嵌入 index.html，正文离线可读。"""

import argparse
import html
import json
import posixpath
import re
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]


def render_note(path):
    source = path.read_text()
    # 先保护代码，再保护数学，避免 Markdown 把 LaTeX 中的下划线或反斜杠当语法。
    codes = []
    def protect_code(match):
        codes.append(match.group())
        return f"CODEPLACEHOLDER{len(codes)-1}TOKEN"
    source = re.sub(r"```.*?```|`[^`\n]+`", protect_code, source, flags=re.S)
    maths = []
    def protect_math(match):
        block = match.group(1) is not None
        body = match.group(1) if block else match.group(2)
        formula = html.escape(body.strip())
        maths.append(
            f'<div class="math-block">\\[{formula}\\]</div>' if block
            else f'<span class="math-inline">\\({formula}\\)</span>'
        )
        return f"MATHPLACEHOLDER{len(maths)-1}TOKEN"
    source = re.sub(r"\$\$(.*?)\$\$|\$([^$\n]+)\$", protect_math, source, flags=re.S)
    for i, code in enumerate(codes):
        source = source.replace(f"CODEPLACEHOLDER{i}TOKEN", code)
    rendered = markdown.markdown(source, extensions=["tables", "fenced_code"])
    for i, formula in enumerate(maths):
        token = f"MATHPLACEHOLDER{i}TOKEN"
        if formula.startswith('<div'):
            rendered = rendered.replace(f"<p>{token}</p>", formula)
        rendered = rendered.replace(token, formula)

    def resolve_link(match):
        target = html.unescape(match.group(1))
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:|/|#", target):
            return match.group()
        return f'href="{html.escape(posixpath.normpath("notes/" + target), quote=True)}"'
    return re.sub(r'href="([^"]+)"', resolve_link, rendered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="仅核对快照是否与源笔记一致")
    args = parser.parse_args()
    notes = {}
    paths = sorted((ROOT / "notes").glob("[0-9][0-9]-*.md"))
    if not paths:
        raise SystemExit("未找到按课号命名的笔记")
    for path in paths:
        note_id = path.name[:2]
        if note_id in notes:
            raise SystemExit(f"第 {note_id} 课存在重复文件")
        notes[note_id] = render_note(path)
    payload = json.dumps(notes, ensure_ascii=False).replace("<", "\\u003c")
    index = ROOT / "index.html"
    original = index.read_text()
    pattern = r'(<script type="application/json" id="notes-content">).*?(</script>)'
    updated, count = re.subn(pattern, lambda m: m[1] + payload + m[2], original, flags=re.S)
    if count != 1:
        raise SystemExit("未找到唯一的笔记快照位置")
    if args.check:
        if updated != original:
            raise SystemExit("笔记快照已过期，请运行 python scripts/build_notes.py")
        print(f"共 {len(notes)} 课网页快照与 Markdown 一致")
    else:
        index.write_text(updated)
        print(f"已更新 index.html 中共 {len(notes)} 课的完整笔记")


if __name__ == "__main__":
    main()
