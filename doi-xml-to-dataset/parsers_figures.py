from __future__ import annotations
from bs4 import BeautifulSoup
from pathlib import Path
import csv, httpx

def download_figures(xml_text: str, out_fig_dir: Path, headers: dict) -> None:
    soup = BeautifulSoup(xml_text, "xml")
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, headers=headers) as client:
        for fig in soup.find_all("ce:figure"):
            link = fig.find("ce:link")
            if not link or not link.get("locator"): 
                continue
            locator = link["locator"]
            obj = soup.find("object", {"ref": locator})
            if not obj or not obj.string: 
                continue
            url = obj.string.strip()
            r = client.get(url)
            if r.status_code == 200:
                (out_fig_dir / f"{locator}.jpg").write_bytes(r.content)

def write_figures_metadata(xml_text: str, out_csv: Path, fig_dir: Path) -> None:
    soup = BeautifulSoup(xml_text, "xml")
    rows = [["Figure ID", "Figure Path", "Figure Label", "Figure Caption"]]
    for fig in soup.find_all("ce:figure"):
        fig_id = fig.get("id", "")
        label = fig.find("ce:label").get_text(strip=True) if fig.find("ce:label") else ""
        caption = fig.find("ce:simple-para").get_text(strip=True) if fig.find("ce:simple-para") else ""
        link = fig.find("ce:link")
        locator = link["locator"] if link and link.get("locator") else ""
        fig_path = (fig_dir / f"{locator}.jpg").as_posix() if locator else ""
        rows.append([fig_id, fig_path, label, caption])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
