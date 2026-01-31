from __future__ import annotations
from pathlib import Path
import csv
from bs4 import BeautifulSoup
from mit_table_extractor_upd import TableExtractor

def write_tables_and_metadata(xml_text: str, xml_path: Path, out_tables_dir: Path, out_meta_csv: Path) -> None:
    soup = BeautifulSoup(xml_text, "xml")
    te = TableExtractor()
    doi_node = soup.find("xocs:doi")
    te.doi = doi_node.text if doi_node else ""
    tables, captions, footers, table_ids, table_labels = te.get_xml_tables(str(xml_path))

    out_tables_dir.mkdir(parents=True, exist_ok=True)
    # tables
    for table_list, table_label in zip(tables, table_labels):
        (out_tables_dir / f"{table_label}.csv").write_text(
            "\n".join(",".join(row) for row in table_list),
            encoding="utf-8",
        )
    # metadata
    rows = [["Table ID", "Table Path", "Table Label", "Table Caption", "Table Footer"]]
    for tid, tlabel, cap, foot in zip(table_ids, table_labels, captions, footers):
        rows.append([tid, (out_tables_dir / f"{tlabel}.csv").as_posix(), tlabel, cap, foot])
    out_meta_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_meta_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
