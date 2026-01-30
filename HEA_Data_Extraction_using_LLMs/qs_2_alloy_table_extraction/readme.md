# LLM-Based Table Property Extraction for High-Entropy Alloys

This repository contains the **Table Extraction** pipeline used to build a structured materials-property database from tables in high-entropy / multicomponent alloy literature.

The code implements a scalable workflow to extract **materials properties from tables** in journal articles using **LLMs + RAG + FAISS**.


---

## 1. Overview

Given a directory of parsed articles (one folder per DOI) containing:

- `Table_Metadata.csv` (caption, footer, table labels)
- `tables/<TableLabel>.csv` (machine-readable tables)

the pipeline:

1. Uses GPT-4o to detect **property-like cells** from the table caption + data + footer.
2. Uses a **FAISS-backed vector store** to retrieve the most relevant **material property definitions** (RAG) from `final_properties_with_descriptions_utf8.csv`.
3. Builds a **few-shot, schema-constrained prompt** and calls either GPT-4o or GPT-4o-mini to:
   - Identify which properties (from a predefined list) are present in the table
   - Extract them into a strict **CSV schema**:

   ```text
   Alloy,Processing condition,Testing condition,Property,Value,Unit
   ```

4. Aggregates all extracted rows into a single output CSV and logs per-table processing details.

---

## 2. Repository Structure

A typical layout is:

```text
.
├─ src/
│  ├─ main.py                        # CLI entrypoint 
│  ├─ vector_store.ipynb                # FAISS index + embedding helpers (optional split)
│  ├─ QuerySet2Postprocessing.ipynb                  # process_table / process_doi_folder / process_dois (optional split)
│  ├─ vector_store.index             # FAISS index (usually .gitignored)
│  └─ metadata.pkl  
├─ data/
│  └─ extracted_data_dois.txt                      # list of DOIs (one per line)
├─ articles_all/
│  └─ <doi-folder>/
│      ├─ Table_Metadata.csv
│      └─ tables/
│          ├─ Table1.csv
│          ├─ Table2.csv
│          └─ ...
├─ QuerySet2_Database.xlsx  # output
├─ requirements.txt
├─ README.md
└─ .gitignore
```
---

## 3. Installation

### 3.1. Clone and create a virtual environment

```bash
git clone https://github.com/<your-username>/he-alloy-table-extraction.git
cd he-alloy-table-extraction

python -m venv .venv
source .venv/bin/activate    
```

### 3.2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. OpenAI API configuration

This project uses the **OpenAI Python SDK (v1+)** and the following models:

- `gpt-4o` – used for:
  - property-name detection from table text
  - complex / large tables
- `gpt-4o-mini` – used for:
  - smaller tables to reduce cost
- `text-embedding-3-large` – used for:
  - building the FAISS vector store over property descriptions

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="...your-key..."
```

On Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY="...your-key..."
```

---

## 5. Data preparation

### 5.1. Property descriptions

Place `final_properties_with_descriptions_utf8.csv` under `data/`.  
This file should include at least:

```text
Property,Description
Yield Strength,"..."
Ultimate Tensile Strength,"..."
...
```

Each row describes a property extracted from materials science literature.  
These descriptions are embedded into the vector store and form the **RAG context**.

### 5.2. Parsed article folders

Expected layout for each DOI folder (under `articles_all/`):

```text
articles_all/
└─ <doi-folder>/
   ├─ Table_Metadata.csv
   └─ tables/
       ├─ Table1.csv
       ├─ Table2.csv
       └─ ...
```

- `Table_Metadata.csv` must at least contain:
  - `Table Label`
  - `Table Caption`
  - `Table Footer`
- Each `tables/<TableLabel>.csv` is the machine-readable table.

### 5.3. DOI list

`data/extracted_data_dois.txt` contains the DOIs (or folder IDs), one per line, in the same order used to create the folders under `articles_all/`.

Example:

```text
./10.1016/j.msea.2008.08.025
./10.1016/j.jmrt.2023.11.138
...
```

The helper `doi_to_foldername` in the code strips the leading `./`.

---

## 6. Running the pipeline

The main entrypoint is `src/main.py`. Example:

```bash
python -m src.main   --root-folder articles_all   --input-file data/extracted_data_dois.txt   --start-index 9373   --end-index 11000   --output-csv final_outputs_table2_rag_v5.csv
```

CLI arguments:

- `--root-folder`  
  Root directory containing per-DOI subfolders (default: `articles_all`)

- `--input-file`  
  Path to the DOIs list (default: `extracted_data_dois.txt`)

- `--start-index` / `--end-index`  
  0-based slice into the DOI list for batching large jobs.  
  For example, `start=0, end=100` processes the first 100 DOIs.

- `--output-csv`  
  File where all extracted rows are appended. If the file already exists,
  the script checks that the header matches before appending.

During execution, the script will:

- Create `query_set2_dois_processing_log_results/<doi-folder>/processing_log.txt`
- Append successful DOIs to `successful_dois_log.txt`
- Log multi-attempt tables with column mismatches in `multi_attempts_log.txt`

---

## 7. How the pipeline works (high level)

For each table:

1. **Read table & metadata**  
   Load `tables/<TableLabel>.csv` + caption/footer from `Table_Metadata.csv`.

2. **Normalize symbols**  
   Convert Greek + non-ASCII characters to ASCII equivalents (`σᵧ → sigma_y`, `ε̇ → epsilon_dot`, etc.) so the embedding model and LLM handle them consistently.

3. **Property-cell detection (GPT-4o)**  
   Pass caption + table data + footer to a **system-prompted GPT-4o** that returns cells it believes contain property names, in a simple comma-separated format.

4. **RAG over property descriptions (FAISS)**  
   Embed each candidate string and query the vector store built on `final_properties_with_descriptions_utf8.csv` to retrieve the top-K most similar property definitions.

5. **Schema-constrained extraction (GPT-4o / GPT-4o-mini)**  
   Construct a detailed prompt that:
   - Shows retrieved property descriptions (context)
   - Provides examples (few-shot) of how to map tables to the target CSV schema
   - Enforces:
     - Exactly **6 columns** per row
     - No commas inside cells (use `;` inside a cell)
     - Only known properties (or `Others(<name>)`)

6. **Validation + retries**  
   Check that all rows in the LLM output have exactly 6 comma-separated fields.  
   If not, retry once, and finally log failures.

7. **Aggregation**  
   Append valid rows into the final output CSV.

---

## 8. Logs and debugging

Each DOI gets its own log file:

```text
query_set2_dois_processing_log_results/
└─ <doi-folder>/
   └─ processing_log.txt
```

This log contains:

- Full input table
- Retrieved properties (from FAISS)
- Final LLM CSV output
- Processing time
- Approximate prompt length (proxy for token usage)

Global logs:

- `successful_dois_log.txt` – lines of `index<TAB>doi` for DOIs with at least one successful table.
- `multi_attempts_log.txt` – DOIs where a table needed multiple attempts due to column mismatch.

Use these files to:

- Inspect where the LLM struggled (e.g., messy tables)
- Filter out problematic DOIs for manual cleaning or error analysis

---

## 9. Reproducing the vector store

If you do not commit `models/vector_store.index` and `models/metadata.pkl`, you can rebuild the FAISS index as follows:

1. Make sure `data/final_properties_with_descriptions_utf8.csv` is present.
2. Add a small script to:
   - Read the CSV
   - Build a list of property strings
   - Call `VectorStore.add_texts(texts)`

Example:

```python
from src.vector_store import VectorStore
import pandas as pd

df = pd.read_csv("data/final_properties_with_descriptions_utf8.csv")
texts = df["Property"].tolist()

store = VectorStore(
    embedding_dim=1536,
    index_file="models/vector_store.index",
    metadata_file="models/metadata.pkl",
)
store.add_texts(texts)
```

You only need to do this once per property-list version.


