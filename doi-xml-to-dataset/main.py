#!/usr/bin/env python3
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from pathlib import Path
from api_elsevier import fetch_xml_by_doi
from parsers_sections import write_sections_csv
from parsers_figures import download_figures, write_figures_metadata
from parsers_tables import write_tables_and_metadata
import os

def doi_to_dir(doi: str) -> str:
    return doi.replace("/", ".")

def run_one(
    doi: str,
    out_root: Path,
    download_figs: bool = True,
    api_key: str | None = None,
    inst_token: str | None = None,
) -> bool:
    xml = fetch_xml_by_doi(doi, api_key=api_key, inst_token=inst_token)
    if not xml:
        return False

    out_dir = out_root / doi_to_dir(doi)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{doi_to_dir(doi)}.xml"
    xml_path.write_text(xml, encoding="utf-8")

    # sections
    write_sections_csv(xml, out_dir / "Processed_Text_Data.csv")

    # figures
    if download_figs:
        headers = {
            "X-ELS-APIKey": api_key or os.getenv("ELS_API_KEY", ""),
            "X-ELS-Insttoken": inst_token or os.getenv("ELS_INST_TOKEN", ""),
            "Accept": "text/xml",
        }
        figs_dir = out_dir / "figures"
        download_figures(xml, figs_dir, headers=headers)
        write_figures_metadata(xml, out_dir / "Figures_Metadata.csv", figs_dir)

    # tables
    write_tables_and_metadata(xml, xml_path, out_dir / "tables", out_dir / "Table_Metadata.csv")
    return True



def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def run_single(doi: str, out_root: Path, download_figs: bool) -> bool:
    ok = run_one(doi, out_root, download_figs=download_figs)
    print("OK" if ok else "FAIL", ":", doi)
    return ok


def run_batch(input_file: Path, out_root: Path, download_figs: bool) -> int:
    lines = input_file.read_text(encoding="utf-8").splitlines()
    total = 0
    for raw in lines:
        doi = raw.strip()
        if not doi:
            continue
        total += int(run_single(doi, out_root, download_figs))
    print(f"Processed {total}/{len([l for l in lines if l.strip()])}")
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract XML → text/figures/tables by DOI (Elsevier API)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--doi", help="Single DOI to process")
    g.add_argument("--input", type=Path, help="Path to a text file with one DOI per line")

    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("articles_all"),
        help="Output root folder (default: articles_all)",
    )
    p.add_argument(
        "--no-figs",
        action="store_true",
        help="Do not download figure images (still writes figure metadata CSV).",
    )
    p.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    return p.parse_args()


# # Single DOI
# python main.py --doi "10.1016/j.msea.2008.08.025"

# # Batch (one DOI per line)
# python main.py --input examples/dois.txt

# # Custom output folder and skip images
# python main.py --doi "10.1016/j.jmrt.2023.11.138" --out-root data_out --no-figs

def main() -> int:
    args = parse_args()
    setup_logging(getattr(logging, args.loglevel.upper(), logging.INFO))
    download_figs = not args.no_figs

    if args.doi:
        ok = run_single(args.doi, args.out_root, download_figs)
        return 0 if ok else 1

    # batch
    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2
    total_ok = run_batch(args.input, args.out_root, download_figs)
    return 0 if total_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
