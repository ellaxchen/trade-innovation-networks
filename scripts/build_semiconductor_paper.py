"""
Build the full semiconductor paper from the front matter + sections markdown.

Outputs:
  ../report/semiconductor_full_paper.md    assembled source (numbered citations)
  ../report/semiconductor_paper.html       self-contained HTML (figures embedded)
  ../report/semiconductor_full_paper.docx  Word draft (TNR 12, double-spaced, 2.54 cm)

Citation keys [@key; @key2] are numbered in order of first appearance and
replaced by Vancouver-style [n] / [n–m]. The reference list is emitted in the
same order.

Usage: python build_semiconductor_paper.py
"""
import base64
import os
import re

import markdown
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

FRONT = "../report/semiconductor_paper_front.md"
SECTIONS = "../report/semiconductor_paper_sections_3-5.md"
OUT_MD = "../report/semiconductor_full_paper.md"
OUT_HTML = "../report/semiconductor_paper.html"
OUT_DOCX = "../report/semiconductor_full_paper.docx"

TITLE = ("Goods Before Ideas? Trade and Knowledge Leadership in "
         "Semiconductors, 1995–2013")
BYLINE = ("Working draft · 31 October 2026 · BACI (HS 8541+8542) · "
          "PATSTAT spring-2018 (CPC H01L) · DOCDB family level · "
          "filing-office baseline")

REFS = {
    "langlois1999": ("Langlois RN, Steinmueller WE. The evolution of competitive "
                     "advantage in the worldwide semiconductor industry, 1947–1996. "
                     "In: Mowery DC, Nelson RR, editors. Sources of Industrial "
                     "Leadership: Studies of Seven Industries. Cambridge: Cambridge "
                     "University Press; 1999. p. 19–78."),
    "brown2009": ("Brown C, Linden G. Chips and Change: How Crisis Reshapes the "
                  "Semiconductor Industry. Cambridge, MA: MIT Press; 2009."),
    "ernst2005": ("Ernst D. Complexity and internationalisation of innovation—why "
                  "is chip design moving to Asia? Int J Innov Manag. "
                  "2005;9(1):47–73."),
    "henderson1990": ("Henderson RM, Clark KB. Architectural innovation: the "
                      "reconfiguration of existing product technologies and the "
                      "failure of established firms. Adm Sci Q. 1990;35(1):9–30."),
    "rauch1999": ("Rauch JE. Networks versus markets in international trade. "
                  "J Int Econ. 1999;48(1):7–35."),
    "chaney2014": ("Chaney T. The network structure of international trade. "
                   "Am Econ Rev. 2014;104(11):3600–3634."),
    "bernard2012": ("Bernard AB, Jensen JB, Redding SJ, Schott PK. The empirics of "
                    "firm heterogeneity in international trade. Annu Rev Econ. "
                    "2012;4:283–313."),
    "antra2013": ("Antràs P, Chor D. Organizing the global value chain. "
                  "Econometrica. 2013;81(6):2121–2204."),
    "baldwin2016": ("Baldwin R. The Great Convergence: Information Technology and "
                    "the New Globalization. Cambridge, MA: Harvard University "
                    "Press; 2016."),
    "griliches1990": ("Griliches Z. Patent statistics as economic indicators: a "
                      "survey. J Econ Lit. 1990;28(4):1661–1707."),
    "jaffe1993": ("Jaffe AB, Trajtenberg M, Henderson R. Geographic localization of "
                  "knowledge spillovers as evidenced by patent citations. "
                  "Q J Econ. 1993;108(3):577–598."),
    "hall2001": ("Hall BH, Jaffe AB, Trajtenberg M. The NBER patent citation data "
                 "file: lessons, insights and methodological tools. NBER Working "
                 "Paper No. 8498. Cambridge, MA: National Bureau of Economic "
                 "Research; 2001."),
    "hidalgo2009": ("Hidalgo CA, Hausmann R. The building blocks of economic "
                    "complexity. Proc Natl Acad Sci USA. 2009;106(26):10570–10575."),
    "hausmann2014": ("Hausmann R, Hidalgo CA, Bustos S, Coscia M, Simoes A, "
                     "Yildirim MA. The Atlas of Economic Complexity: Mapping Paths "
                     "to Prosperity. Cambridge, MA: MIT Press; 2014."),
    "pavitt1984": ("Pavitt K. Sectoral patterns of technical change: towards a "
                   "taxonomy and a theory. Res Policy. 1984;13(6):343–373."),
    "patel1997": ("Patel P, Pavitt K. The technological competencies of the world's "
                  "largest firms: complex and path-dependent, but not much variety. "
                  "Res Policy. 1997;26(2):141–156."),
    "gaulier2010": ("Gaulier G, Zignago S. BACI: International Trade Database at the "
                    "Product-Level. CEPII Working Paper No. 2010-23. Paris: CEPII; "
                    "2010."),
    "epo2018": ("European Patent Office. EPO Worldwide Patent Statistical Database "
                "(PATSTAT). Spring 2018 edition. Munich: European Patent Office; "
                "2018."),
    "lybbert2014": ("Lybbert TJ, Zolas NJ. Getting patents and economic data to "
                    "speak to each other: an 'algorithmic links with probabilities' "
                    "approach for joint analyses of patenting and economic activity. "
                    "Res Policy. 2014;43(3):530–542."),
    "brin1998": ("Brin S, Page L. The anatomy of a large-scale hypertextual web "
                 "search engine. Comput Netw ISDN Syst. 1998;30(1–7):107–117."),
    "freeman1978": ("Freeman LC. Centrality in social networks: conceptual "
                    "clarification. Soc Networks. 1978;1(3):215–239."),
    "barabasi2016": ("Barabási AL. Network Science. Cambridge: Cambridge University "
                     "Press; 2016."),
    "sfj2023": ("Southern Finance Journal. Article sfj-2023-004 [full bibliographic "
                "details to be completed from the journal]."),
}

CITATION_RE = re.compile(r"\[([^\[\]]*@[^\[\]]*)\]")

CHORD_IFRAME_HTML = """
<div class="chord-embed">
<iframe src="../html/chord_semiconductor_standalone.html"
        title="Interactive semiconductor chord diagram"></iframe>
<p class="small">Interactive: use the year slider and country selector.
Inventor-country version:
<a href="../html/chord_semiconductor_inventor_standalone.html">open here</a>.</p>
</div>
"""


def fmt_numbers(nums):
    nums = sorted(set(nums))
    parts = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"{nums[i]}–{nums[j]}")
        else:
            parts.extend(str(n) for n in nums[i:j + 1])
        i = j + 1
    return ",".join(parts)


def number_citations(text):
    order = []

    def repl(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for key in keys:
            if key not in REFS:
                raise KeyError(f"Citation key not in reference table: {key}")
            if key not in order:
                order.append(key)
            nums.append(order.index(key) + 1)
        return f"[{fmt_numbers(nums)}]"

    return CITATION_RE.sub(repl, text), order


def assemble():
    with open(FRONT, encoding="utf-8") as f:
        front = f.read()
    with open(SECTIONS, encoding="utf-8") as f:
        sections = f.read()
    text = front.rstrip() + "\n\n" + sections.lstrip()
    text, order = number_citations(text)

    unused = [k for k in REFS if k not in order]
    if unused:
        print(f"  NOTE: reference keys not cited: {', '.join(unused)}")
    refs_md = "# References\n\n" + "\n".join(
        f"{i}. {REFS[k]}" for i, k in enumerate(order, 1))
    if "<!--REFERENCES-->" not in text:
        raise RuntimeError("Missing <!--REFERENCES--> placeholder")
    text = text.replace("<!--REFERENCES-->", refs_md)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"  Saved -> {OUT_MD}")
    return text, order


# ---------------------------------------------------------------
# HTML
# ---------------------------------------------------------------
HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TITLE_TEXT</title>
<style>
:root{
  --surface-1:#fcfcfb; --page:#f9f9f7; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --muted:#898781; --grid:#e1e0d9;
  --baseline:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Georgia,"Times New Roman",serif;background:var(--page);
     color:var(--text-primary);line-height:1.62;font-size:16.5px}
.paper{max-width:900px;margin:0 auto;padding:52px 30px 100px}
h1{font-size:1.32rem;font-weight:700;letter-spacing:-0.01em;
   line-height:1.3;margin:36px 0 10px;padding-top:10px;
   border-top:1px solid var(--grid)}
.paper > h1:first-child{font-size:1.7rem;margin:0 0 8px;padding:0;
                        border:none;line-height:1.26}
.byline{color:var(--text-secondary);font-size:0.86rem;margin-bottom:28px;
        font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
h2{font-size:1.14rem;font-weight:700;margin:22px 0 6px}
h3{font-size:1.03rem;font-weight:700;margin:20px 0 6px}
p{margin:9px 0;text-align:justify;hyphens:auto}
a{color:var(--series-1);text-decoration:none}
a:hover{text-decoration:underline}
.abstract{background:var(--surface-1);border:1px solid var(--border);
          border-radius:12px;padding:18px 24px;margin:14px 0 22px;
          font-size:0.92rem}
.abstract h2{border:none;margin:0 0 8px;padding:0;font-size:0.82rem;
             text-transform:uppercase;letter-spacing:.07em;
             color:var(--text-secondary)}
.abstract p{text-align:justify}
figure{background:#fff;border:1px solid var(--border);border-radius:10px;
       padding:10px;margin:20px 0 8px;overflow:hidden}
figure img{width:100%;height:auto;display:block;border-radius:6px}
.chord-embed{margin:16px 0}
.chord-embed iframe{width:100%;height:720px;border:1px solid var(--border);
                    border-radius:10px;background:#1a1a2e;display:block}
figcaption{font-size:0.8rem;color:var(--text-secondary);margin-top:8px;
           text-align:left}
table{border-collapse:collapse;width:100%;font-size:0.84rem;margin:12px 0;
      font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600;font-size:0.76rem;
   border-bottom:2px solid var(--baseline)}
ol{padding-left:1.7em;margin:8px 0}
ol li{margin:5px 0;font-size:0.86rem;text-align:left}
code{font-family:"SFMono-Regular",Consolas,monospace;font-size:0.86em;
     background:rgba(11,11,11,0.05);padding:1px 4px;border-radius:4px}
hr{border:none;border-top:1px solid var(--grid);margin:26px 0}
.small{font-size:0.84rem;color:var(--text-secondary)}
@media print{
  body{background:#fff;font-size:11.5pt}
  .paper{max-width:none;padding:0}
  h1{font-size:16pt} h2{font-size:13pt} h3{font-size:11.5pt}
  h2,h3{page-break-after:avoid}
  figure,table{page-break-inside:avoid}
  a{color:inherit;text-decoration:none}
}
</style>
</head>
<body>
<div class="paper">
<h1>TITLE_TEXT</h1>
<div class="byline">BYLINE_TEXT</div>
"""

HTML_TAIL = """
</div>
</body>
</html>
"""


def embed_figures(html):
    def repl(m):
        block = m.group(0)
        paths = re.findall(r"<code>(\.\./plots/[^<]+\.png)</code>", block)
        if not paths:
            return block
        path = paths[0]
        full = os.path.normpath(os.path.join("..", "report", path))
        if not os.path.exists(full):
            print(f"  WARNING: missing figure {full}")
            return block
        with open(full, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        label_m = re.search(r"<strong>(Figure \d+)</strong>", block)
        label = label_m.group(1) if label_m else "Figure"
        cleaned = re.sub(r"\([^)]*<code>[^<]+\.png</code>[^)]*\)", "", block)
        cleaned = re.sub(r"<code>[^<]+\.png</code>", "", cleaned)
        cleaned = cleaned.replace("()", "")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.replace("<p> ", "<p>").replace(" </p>", "</p>")
        return (f'<figure><img src="data:image/png;base64,{b64}" '
                f'alt="{label}"></figure>\n{cleaned}')

    return re.sub(r"<p>.*?</p>", repl, html, flags=re.S)


def build_html(md_text):
    md_for_html = md_text.replace("<!--CHORD_IFRAME-->", CHORD_IFRAME_HTML)
    body = markdown.markdown(md_for_html, extensions=["tables"])
    # wrap the abstract
    body = re.sub(
        r'(<h2>Abstract</h2>.*?)(?=<h1>1\. Introduction</h1>)',
        r'<div class="abstract">\1</div>', body, flags=re.S)
    body = embed_figures(body)
    html = (HTML_HEAD.replace("TITLE_TEXT", TITLE)
            .replace("BYLINE_TEXT", BYLINE) + body + HTML_TAIL)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved -> {OUT_HTML} ({os.path.getsize(OUT_HTML)/1024:.0f} KB)")


# ---------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------
def add_runs(paragraph, text):
    tokens = re.split(r"(\*\*.*?\*\*|\*.*?\*)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = paragraph.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = paragraph.add_run(tok[1:-1]); r.italic = True
        else:
            paragraph.add_run(tok)


def clean_inline(text):
    text = text.replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def add_page_number(document):
    footer = document.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


def build_docx(md_text):
    doc = Document()
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

    for name, size in [("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)]:
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        st.paragraph_format.space_before = Pt(12)
        st.paragraph_format.space_after = Pt(6)

    # title + byline
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    r = p.add_run(TITLE); r.bold = True; r.font.size = Pt(16)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r = p.add_run(BYLINE); r.italic = True; r.font.size = Pt(10)
    doc.add_paragraph()

    blocks = [b.strip() for b in re.split(r"\n\s*\n", md_text) if b.strip()]
    for block in blocks:
        # headings
        if block.startswith("#"):
            level = len(block) - len(block.lstrip("#"))
            text = block.lstrip("#").strip()
            doc.add_heading(clean_inline(text), level=min(level, 3))
            continue

        # interactive chord placeholder (HTML only; DOCX gets a path note)
        if "<!--CHORD_IFRAME-->" in block:
            p = doc.add_paragraph()
            r = p.add_run(
                "Interactive HTML versions: html/chord_semiconductor_standalone.html "
                "and html/chord_semiconductor_inventor_standalone.html.")
            r.italic = True
            continue

        # tables
        if block.startswith("|"):
            lines = [ln.strip().strip("|") for ln in block.splitlines()
                     if ln.strip()]
            rows = []
            for ln in lines:
                core = ln.replace("|", "").replace("-", "").replace(":", "").strip()
                if set(core) == set():
                    continue
                rows.append([c.strip() for c in ln.split("|")])
            if not rows:
                continue
            ncols = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=ncols)
            table.style = "Table Grid"
            for ri, row in enumerate(rows):
                for ci, val in enumerate(row):
                    cell = table.cell(ri, ci)
                    cell.text = ""
                    cp = cell.paragraphs[0]
                    cp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                    add_runs(cp, clean_inline(val))
                    for run in cp.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(9)
                        if ri == 0:
                            run.bold = True
            doc.add_paragraph()
            continue

        # figure paragraphs
        fig_paths = re.findall(r"`([^`]*\.png)`", block)
        if fig_paths:
            path = fig_paths[0]
            full = os.path.normpath(os.path.join("..", "report", path))
            caption = clean_inline(block)
            caption = re.sub(r"\([^)]*\.png[^)]*\)", "", caption)
            caption = re.sub(r"\.\./plots/[^\s`)]+\.png", "", caption)
            caption = caption.replace("()", "")
            caption = re.sub(r"\s+", " ", caption).strip()
            if os.path.exists(full):
                try:
                    doc.add_picture(full, width=Inches(6.3))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                except Exception as exc:
                    doc.add_paragraph(f"[Figure missing: {full} ({exc})]")
            else:
                doc.add_paragraph(f"[Figure missing: {full}]")
            cap = doc.add_paragraph()
            cap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            add_runs(cap, caption)
            for run in cap.runs:
                run.font.size = Pt(10)
            doc.add_paragraph()
            continue

        # ordered lists (references)
        if all(re.match(r"^\d+\.\s", ln) for ln in block.splitlines()):
            for ln in block.splitlines():
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.35)
                p.paragraph_format.first_line_indent = Inches(-0.35)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
                add_runs(p, clean_inline(ln))
            continue

        # ordinary paragraph
        p = doc.add_paragraph()
        add_runs(p, clean_inline(block))

    add_page_number(doc)
    doc.save(OUT_DOCX)
    print(f"  Saved -> {OUT_DOCX} ({os.path.getsize(OUT_DOCX)/1024:.0f} KB)")


if __name__ == "__main__":
    print("[1/3] Assembling markdown + numbering citations...")
    full_md, order = assemble()
    print(f"  {len(order)} references, numbered in citation order")
    print("[2/3] Building HTML...")
    build_html(full_md)
    print("[3/3] Building DOCX...")
    build_docx(full_md)
    print("Done.")
