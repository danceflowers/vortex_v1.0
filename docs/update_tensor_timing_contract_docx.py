from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
ET.register_namespace("w", W)

MD_PATH = Path("docs/tensor_timing_contract.md")
DOCX_PATH = Path("docs/tensor_timing_contract.docx")
UPDATED_DOCX_PATH = Path("docs/tensor_timing_contract.updated.docx")


def w(tag: str) -> str:
    return f"{{{W}}}{tag}"


def make_paragraph(text: str, style: str | None = None, bold: bool = False) -> ET.Element:
    p = ET.Element(w("p"))
    if style is not None:
        ppr = ET.SubElement(p, w("pPr"))
        ET.SubElement(ppr, w("pStyle"), {w("val"): style})
    r = ET.SubElement(p, w("r"))
    if bold:
        rpr = ET.SubElement(r, w("rPr"))
        ET.SubElement(rpr, w("b"))
    t = ET.SubElement(r, w("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return p


def make_table(rows: list[list[str]]) -> ET.Element:
    tbl = ET.Element(w("tbl"))
    tbl_pr = ET.SubElement(tbl, w("tblPr"))
    ET.SubElement(tbl_pr, w("tblW"), {w("w"): "0", w("type"): "auto"})
    for row in rows:
        tr = ET.SubElement(tbl, w("tr"))
        for cell in row:
            tc = ET.SubElement(tr, w("tc"))
            p = ET.SubElement(tc, w("p"))
            r = ET.SubElement(p, w("r"))
            t = ET.SubElement(r, w("t"))
            t.text = cell
    return tbl


def parse_markdown(lines: list[str]) -> list[ET.Element]:
    blocks: list[ET.Element] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows: list[list[str]] = []
            for idx, raw in enumerate(table_lines):
                if idx == 1 and re.fullmatch(r"\|[\-\s|:]+\|", raw):
                    continue
                parts = [c.strip() for c in raw.strip("|").split("|")]
                rows.append(parts)
            if rows:
                blocks.append(make_table(rows))
            continue

        if stripped.startswith("# "):
            blocks.append(make_paragraph(stripped[2:].strip(), style="Heading1"))
            i += 1
            continue
        if stripped.startswith("## "):
            blocks.append(make_paragraph(stripped[3:].strip(), style="Heading2"))
            i += 1
            continue
        if stripped.startswith("### "):
            blocks.append(make_paragraph(stripped[4:].strip(), style="Heading3"))
            i += 1
            continue

        blocks.append(make_paragraph(stripped))
        i += 1

    return blocks


def update_docx(md_path: Path, docx_path: Path, fallback_docx_path: Path) -> Path:
    markdown = md_path.read_text(encoding="utf-8").splitlines()
    blocks = parse_markdown(markdown)

    with ZipFile(docx_path, "r") as zin:
        xml = zin.read("word/document.xml")
        others = {name: zin.read(name) for name in zin.namelist() if name != "word/document.xml"}

    root = ET.fromstring(xml)
    body = root.find("w:body", NS)
    assert body is not None

    sect_pr = None
    if len(body) and body[-1].tag == w("sectPr"):
        sect_pr = copy.deepcopy(body[-1])

    for child in list(body):
        body.remove(child)

    for block in blocks:
        body.append(block)

    if sect_pr is not None:
        body.append(sect_pr)

    new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    def write_output(path: Path) -> None:
        with TemporaryDirectory() as td:
            tmp = Path(td) / path.name
            with ZipFile(tmp, "w") as zout:
                for name, data in others.items():
                    zout.writestr(name, data)
                zout.writestr("word/document.xml", new_xml)
            path.write_bytes(tmp.read_bytes())

    try:
        write_output(docx_path)
        return docx_path
    except PermissionError:
        write_output(fallback_docx_path)
        return fallback_docx_path


if __name__ == "__main__":
    out = update_docx(MD_PATH, DOCX_PATH, UPDATED_DOCX_PATH)
    print(out)
