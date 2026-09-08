"""build_katex_css.py — self-contained KaTeX stylesheet.

The artifact CSP allows scripts from cdnjs but blocks external stylesheets and font
files, so KaTeX's normal CSS + webfont path cannot load. This inlines the fonts we
actually use as data: URIs and drops the @font-face rules for the rest, producing one
<style> block with no external requests. Run once; output is committed.

    python build_katex_css.py     # reports/katex/katex_inline.css
"""
import base64
import pathlib
import re

SRC = pathlib.Path("reports/katex")
# faces our equations can reach: roman text, italic variables, and the size-variants
# used by \sqrt, \frac and \left(...\right)
KEEP = ["KaTeX_Main-Regular", "KaTeX_Main-Bold", "KaTeX_Main-Italic",
        "KaTeX_Math-Italic", "KaTeX_Size1-Regular", "KaTeX_Size2-Regular",
        "KaTeX_Size3-Regular", "KaTeX_Size4-Regular", "KaTeX_AMS-Regular"]


def main():
    css = (SRC / "katex.min.css").read_text()
    blocks = re.findall(r"@font-face\{[^}]*\}", css)
    kept, dropped = [], 0
    for b in blocks:
        # the woff2 filename is the unambiguous face name; the font-family/weight
        # declarations are inconsistently quoted and numeric across KaTeX releases
        m = re.search(r"url\(fonts/(KaTeX_[A-Za-z0-9]+-[A-Za-z]+)\.woff2\)", b)
        if not m:
            css = css.replace(b, "")
            dropped += 1
            continue
        name = m.group(1)
        path = SRC / "fonts" / f"{name}.woff2"
        if name not in KEEP or not path.exists():
            css = css.replace(b, "")
            dropped += 1
            continue
        b64 = base64.b64encode(path.read_bytes()).decode()
        css = css.replace(b, re.sub(r"src:[^;}]*", f"src:url(data:font/woff2;base64,{b64}) format('woff2')", b))
        kept.append(name)
    out = SRC / "katex_inline.css"
    out.write_text(css)
    print(f"embedded {len(kept)} faces, dropped {dropped}; {out} = {len(css)/1024:.0f}KB")
    print("  " + ", ".join(sorted(set(kept))))


if __name__ == "__main__":
    main()
