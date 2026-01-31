# doi-xml-to-dataset

Extracts processed text, figures (and metadata), and tables (and metadata) from publisher XML by DOI using the Elsevier Content API.

Outputs a tidy folder per DOI with:
- `Processed_Text_Data.csv` (sections/paragraphs + figure/table refs)
- `Figures_Metadata.csv` (+ optional downloaded images in `figures/`)
- `Table_Metadata.csv` and per-table CSVs under `tables/`

---

## Quick start

### Requirements
- Python **3.10+**
- Elsevier API credentials in environment variables:
  - `ELS_API_KEY`
  - `ELS_INST_TOKEN`

### Install
```bash
# from repo root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r req.txt
```

### Set credentials
```bash
# macOS/Linux
export ELS_API_KEY="your-elsevier-api-key"
export ELS_INST_TOKEN="your-elsevier-inst-token"

# Windows PowerShell
# $env:ELS_API_KEY="your-elsevier-api-key"
# $env:ELS_INST_TOKEN="your-elsevier-inst-token"
```

### Run
```bash
# Single DOI
python main.py --doi "10.1007/s10853-016-0609-x"

# Batch (one DOI per line)
python main.py --input examples/dois.txt

# Custom output folder and skip images
python main.py --doi "10.1016/j.jmrt.2023.11.138" --out-root data_out --no-figs
```

---

## Output layout

For a DOI like `{doi}`, outputs go to `articles_all/{doi}/`:

```
articles_all/
└─ {doi}/
   ├─ {doi}.xml
   ├─ Processed_Text_Data.csv
   ├─ Figures_Metadata.csv
   ├─ figures/               # only if images are downloaded
   │  ├─ f1.jpg
   │  └─ ...
   ├─ Table_Metadata.csv
   └─ tables/
      ├─ Table 1.csv
      ├─ Table 2.csv
      └─ ...
```

**CSV Schemas**
- `Processed_Text_Data.csv`: `Section Heading, Section Title, Paragraph, Figures, Tables`
- `Figures_Metadata.csv`: `Figure ID, Figure Path, Figure Label, Figure Caption`
- `Table_Metadata.csv`: `Table ID, Table Path, Table Label, Table Caption, Table Footer`

---

## Command-line options

```
--doi "DOI"                 # process a single DOI
--input path/to/dois.txt    # process multiple DOIs (one per line)
--out-root PATH             # output root (default: articles_all)
--no-figs                   # do not download figure images (still writes metadata)
--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}  # default: INFO
```

**Exit codes**
- `0` success
- `1` failure (no successful items)
- `2` invalid input (e.g., missing file)

---

## How it works (brief)

1. Fetch article XML via Elsevier Content API.
2. Parse sections/paragraphs to CSV; preserve figure/table cross-refs.
3. Parse figure labels/captions; optionally download figure images and write metadata.
4. Parse tables using the adapted MIT extractor; write per-table CSVs + metadata.

---

## Dependencies

Listed in [`req.txt`](./req.txt). Core libraries:
- `beautifulsoup4`, `lxml` (parsing)
- `httpx` (API calls)
- `rich` (logging)
- `unidecode`, `scipy` (table processing)

---

## Attribution (Tables)

Table extraction is adapted from: <https://github.com/olivettigroup/table_extractor>  
Licensed under **MIT** (license header included in `src/paper_xml_extractor/mit_table_extractor_updated.py`).

---

