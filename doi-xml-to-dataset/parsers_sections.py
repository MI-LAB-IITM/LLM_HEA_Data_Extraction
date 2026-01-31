from __future__ import annotations
from bs4 import BeautifulSoup
from pathlib import Path
import csv, re

_RE_FIGTAB = re.compile(r'(Fig\. \d+\([a-z]\)|Figs\. \d+|Fig\. \d+|Table \d+)')

def _add_spaces(text: str) -> str:
    return _RE_FIGTAB.sub(lambda m: f" {m.group(0)} ", text)

def _extract_section_hierarchy(section, visited: set[str], parent_label: str = "", parent_title: str = ""):
    title_node = section.find("ce:section-title")
    label_node = section.find("ce:label")
    title = title_node.get_text(strip=True) if title_node else ""
    label = label_node.get_text(strip=True) if label_node else ""

    cur_label = f"{label}"
    cur_title = f"{parent_title}.{title}" if parent_title else title

    if cur_label in visited:
        return []

    visited.add(cur_label)
    data = []
    subsections = section.find_all("ce:section", recursive=False)

    if not subsections:
        for para in section.find_all("ce:para"):
            ptxt = _add_spaces(para.get_text(strip=True))
            figures, tables = [], []
            for ref in para.find_all("ce:cross-ref", {"refid": True}):
                refid = ref["refid"].lower()
                if refid.startswith("f"): figures.append(ref["refid"])
                elif refid.startswith("t"): tables.append(ref["refid"])
            for refs in para.find_all("ce:cross-refs", {"refid": True}):
                for rid in refs["refid"].split():
                    rid_l = rid.lower()
                    if rid_l.startswith("f"): figures.append(rid)
                    elif rid_l.startswith("t"): tables.append(rid)
            data.append([cur_label, cur_title, ptxt, figures, tables])
    else:
        for ss in subsections:
            data.extend(_extract_section_hierarchy(ss, visited, cur_label, cur_title))
    return data

def write_sections_csv(xml_text: str, out_csv: Path) -> None:
    soup = BeautifulSoup(xml_text, "xml")
    rows = [["Section Heading", "Section Title", "Paragraph", "Figures", "Tables"]]
    abs_node = soup.find("dc:description")
    abstract = (abs_node.get_text().replace("Abstract", "").replace("\n", "").strip()) if abs_node else ""
    rows.append(["0", "Abstract", abstract, [], []])

    visited: set[str] = set()
    for sec in soup.find_all("ce:section"):
        rows.extend(_extract_section_hierarchy(sec, visited))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
