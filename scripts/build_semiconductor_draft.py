"""
Build the semiconductor draft (.docx) from the markdown sections file.

Formatting follows the SFJ brief: 12-pt Times New Roman, double-spaced,
2.54 cm margins. Figures referenced as `../plots/*.png` in backticks are
embedded at their paragraph position with the paragraph text as the caption.

Usage: python build_semiconductor_draft.py
Input:  ../report/semiconductor_paper_sections_3-5.md
Output: ../report/semiconductor_draft_sections_3-5.docx
"""
import os
import re

from docx import Document
from docx.shared import Pt, Cm, Inches
from docx.enum.text import WD_LINE_SPACING
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC = "../report/semiconductor_paper_sections_3-5.md"
OUT = "../report/semiconductor_draft_sections_3-5.docx"

doc = Document()

# ---- page + base styles ----
for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
normal.paragraph_format.space_after = Pt(0)

for name in ["Heading 1", "Heading 2", "Heading 3"]:
    st = doc.styles[name]
    st.font.name = "Times New Roman"
    st.font.color.rgb = None
    st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    st.paragraph_format.space_before = Pt(12)
    st.paragraph_format.space_after = Pt(6)

doc.styles["Heading 1"].font.size = Pt(14)
doc.styles["Heading 2"].font.size = Pt(12)
doc.styles["Heading 1"].font.bold = True
doc.styles["Heading 2"].font.bold = True


def add_runs(paragraph, text):
    """Add text with **bold** and *italic* markers as runs."""
    tokens = re.split(r"(\*\*.*?\*\*|\*.*?\*)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = paragraph.add_run(tok[2:-2])
            r.bold = True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = paragraph.add_run(tok[1:-1])
            r.italic = True
        else:
            paragraph.add_run(tok)


def clean(text):
    text = text.replace("`", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


with open(SRC, encoding="utf-8") as f:
    raw = f.read()

# split into blocks by blank lines
blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]

i = 0
while i < len(blocks):
    block = blocks[i]

    # headings
    if block.startswith("#"):
        level = len(block) - len(block.lstrip("#"))
        text = block.lstrip("#").strip()
        doc.add_heading(text, level=min(level, 2))
        i += 1
        continue

    # tables
    if block.startswith("|"):
        rows = [r.strip().strip("|") for r in block.splitlines() if r.strip()]
        rows = [r for r in rows if not set(r.replace("|", "").replace("-", "").replace(":", "").strip()) == set()]
        if not rows:
            i += 1
            continue
        cells = [[c.strip() for c in r.split("|")] for r in rows]
        ncols = max(len(r) for r in cells)
        table = doc.add_table(rows=len(cells), cols=ncols)
        table.style = "Table Grid"
        for r_i, row in enumerate(cells):
            for c_i, val in enumerate(row):
                cell = table.cell(r_i, c_i)
                cell.text = ""
                p = cell.paragraphs[0]
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                add_runs(p, clean(val))
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(9)
                    if r_i == 0:
                        run.bold = True
        doc.add_paragraph()
        i += 1
        continue

    # figure paragraphs: text containing a backticked png path
    fig_paths = re.findall(r"`([^`]*\.png)`", block)
    if fig_paths:
        path = fig_paths[0]
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(os.path.dirname(SRC), path))
        caption = clean(block)
        # drop every png reference, including parenthetical asides that
        # contain one, then tidy empty parentheses
        caption = re.sub(r"\([^)]*\.png[^)]*\)", "", caption)
        caption = re.sub(r"\.\./plots/[^\s`)]+\.png", "", caption)
        caption = caption.replace("()", "")
        caption = re.sub(r"\s+", " ", caption).strip()
        if os.path.exists(path):
            try:
                doc.add_picture(path, width=Inches(6.3))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            except Exception as exc:
                doc.add_paragraph(f"[Figure missing: {path} ({exc})]")
        else:
            doc.add_paragraph(f"[Figure missing: {path}]")
        cap = doc.add_paragraph()
        cap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        add_runs(cap, caption)
        for run in cap.runs:
            run.font.size = Pt(10)
        doc.add_paragraph()
        i += 1
        continue

    # horizontal rules are layout only
    if block.strip() in {"---", "***", "___"}:
        i += 1
        continue

    # ordinary paragraph(s)
    text = clean(block)
    p = doc.add_paragraph()
    add_runs(p, text)
    i += 1

doc.save(OUT)
print(f"Saved -> {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")
