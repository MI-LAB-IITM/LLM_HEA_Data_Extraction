# Query Set 1 – Alloy Text Extraction Pipeline

This repository contains the **Query Set 1** pipeline for extracting alloy-related information from research articles using **LLMs + RAG (retrieval-augmented generation)**.

The goal is to build a structured dataset of:

- **Alloy name / composition**
- **Processing conditions**
- **Characterization techniques**
- **Properties**

from the text of materials-science articles (typically multicomponent / high-entropy alloys).

---

## 1. Repository Structure

A suggested layout for the repo is:

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── rag_examples.py
├── embeddings_utils.py
├── openai_clients.py
├── doi_utils.py
├── material_extractor.py
├── run_batch.py
└── ipynb_to_py.py
```

You will also have runtime data/output directories, e.g.:

```text
./articles_all/                      # Input: per-DOI processed text
./query_set1_dois_results_dec_ver_all/  # Output: per-DOI extraction results
./examples_embeddings/               # Precomputed example embeddings (.pkl)
./logs_dec_ver_q1_set.txt            # Global log for batch runs
```

These folders are usually **not** committed fully to Git; you can add them to `.gitignore` or commit a small sample.

---

## 2. Installation

### Python Version

Use **Python 3.9+** (tested with 3.9/3.10/3.11).

### Install Dependencies

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
openai>=1.0.0
httpx==0.27.2
transformers
faiss-cpu
numpy
pandas
nbformat
```

Add extra libraries if needed (e.g. `tqdm`).

---

## 3. OpenAI Configuration

All OpenAI API keys and model names are configured via environment variables.

### 3.1. `.env.example`

An example `.env.example` file:

```bash
# Copy this file to `.env` and fill in your values.
# Then you can either `export` them manually or use a loader like python-dotenv.

# Default / fallback API key
OPENAI_API_KEY="your_default_openai_key"

# Optional: separate keys per use-case
OPENAI_API_KEY_EMBED="your_key_for_embeddings"
OPENAI_API_KEY_GPT4O="your_key_for_gpt4o"
OPENAI_API_KEY_GPT4O_MINI="your_key_for_gpt4o_mini"

# Optional model overrides
EMBED_MODEL="text-embedding-3-small"
CHAT_MODEL_STRONG="gpt-4o"
CHAT_MODEL_LIGHT="gpt-4o-mini"

# Optional path overrides
ARTICLES_ROOT="./articles_all"
RESULTS_ROOT="./query_set1_dois_results_dec_ver_all"
EXAMPLES_EMBEDDINGS_DIR="./examples_embeddings"
DOIS_FILE="./extracted_data_dois.txt"
GLOBAL_LOG_FILE="./logs_dec_ver_q1_set.txt"
```

Create a real `.env` (or export variables in your shell) with your keys.

> **Important:** Never commit actual API keys to the repository.

---

## 4. Core Modules

### 4.1. `openai_clients.py`

Centralizes all OpenAI client creation and chat/embedding wrappers.

Key parts:

- Reads keys from `OPENAI_API_KEY_*` or `OPENAI_API_KEY`.
- `get_embedding(text, model=EMBED_MODEL)`
- `generate_chat_completion(prompt, mode="strong" | "light", ...)` with
  - `mode="strong"` → typically GPT‑4o (for harder tasks)
  - `mode="light"` → GPT‑4o‑mini (for cheaper calls)
- Implements **exponential backoff** on API errors.

You can switch models just by changing environment variables.

---

### 4.2. `rag_examples.py`

Contains the **few-shot examples** used in the prompts.

There are four main lists:

- `alloy_examples`
- `processing_condition_examples`
- `characterization_examples`
- `properties_examples`

Each entry is of the form:

```text
"Article: ...\nAnswer: ..."
```

These examples are embedded and used as a mini RAG store to condition the LLM with relevant patterns for each paragraph and task.

---

### 4.3. `embeddings_utils.py`

Utility functions for computing and using embeddings via FAISS.

Main functions:

- `preprocess_text(text) -> np.ndarray`
  - Calls `get_embedding` from `openai_clients.py` and returns a float32 numpy array.

- `precompute_and_save_embeddings(example_texts, filename)`
  - Computes embeddings for a list of example strings and saves them to a `.pkl` file under `./examples_embeddings/`.

- `load_embeddings(filename)`
  - Loads a `.pkl` embedding file.

- `find_similar_examples(example_embeddings, user_paragraph, examples, k)`
  - Builds a FAISS `IndexFlatL2` and finds the **top‑k** nearest example indices.
  - Returns a tuple `(found_special, top_examples)`:
    - `found_special`: whether any of indices `{24, 25}` appears (your original heuristic).
    - `top_examples`: the top‑k example strings.

- `find_similar_examples_from_precomputed(user_paragraph, embedding_file, examples, k)`
  - Convenience wrapper: loads embeddings from file then calls `find_similar_examples`.

These helpers are used to select **relevant few-shot examples** for each paragraph and task type (alloy extraction, processing conditions, etc.).

---

### 4.4. `doi_utils.py`

Helper functions for working with DOIs and DOI lists.

- `save_dois_to_file(dois_list, output_file)`
  - Writes one DOI per line.

- `read_dois_from_file(input_file)`
  - Reads all DOIs from a text file, stripping whitespace.

- `doi_to_foldername(doi)`
  - Converts a DOI to a folder-friendly name (currently simply `replace("./", "")`).

The folder name is used under `ARTICLES_ROOT` and `RESULTS_ROOT` as the per‑DOI directory.

---

### 4.5. `material_extractor.py`

This is the **core Query Set 1 logic**.

#### Inputs

- Environment variables:
  - `ARTICLES_ROOT` (default `./articles_all`)
  - `RESULTS_ROOT` (default `./query_set1_dois_results_dec_ver_all`)
  - `EXAMPLES_EMBEDDINGS_DIR` (default `./examples_embeddings`)

- For each DOI:
  - A folder under `ARTICLES_ROOT`, e.g. `./articles_all/<folder_name>/`
  - Inside it: `Processed_Text_Data.csv` with columns:
    - `Section Heading` (e.g. `0`, `2`, `2.1`, ...)
    - `Section Title`
    - `Paragraph`

Only rows whose `Section Heading` starts with `'0'` or `'2'` are used (typically intro + experimental sections).

#### Pipeline Overview

For a given DOI:

1. **Locate input file**: `ARTICLES_ROOT/<folder_name>/Processed_Text_Data.csv`.
2. **Skip** if input folder missing or if output folder (`RESULTS_ROOT/<folder_name>/`) already exists.
3. **Filter sections**: `Section Heading` starting with `0` or `2`.
4. **Step 1 – Extract alloys**:
   - For each paragraph:
     - Retrieve top‑k `alloy_examples` via FAISS.
     - Build a long, instruction‑rich prompt covering:
       - What an alloy is
       - How to interpret notations like `AlxCoCrFeNi`, `Alx(TiVCr)100−x`, etc.
       - The retrieved few-shot examples
     - Query the LLM with:
       - Strong model (GPT‑4o) if `found_special=True` from FAISS
       - Light model (GPT‑4o‑mini) otherwise
     - The model response is expected as comma‑separated alloy compositions or `-`.
     - Valid alloys are collected into a `set(all_alloys)`.

5. **Step 2 – For each alloy, extract:**
   - **Processing conditions**
   - **Characterization techniques**
   - **Properties**

   For each alloy and each paragraph:
   - Retrieve top‑k examples from the corresponding example list (`processing_condition_examples`, `characterization_examples`, `properties_examples`).
   - Build a task-specific prompt with instructions + examples.
   - Use the **light** model (GPT‑4o‑mini) to respond.
   - Clean the response to ASCII (strip non‑ASCII characters).
   - Append to a row of the form:

     ```text
     [Alloy name, Processing conditions, Characterization techniques, Properties]
     ```

6. **Token / character statistics**
   - Track number of prompts and approximate character counts separately for strong vs light model.

7. **Outputs per DOI** (under `RESULTS_ROOT/<folder_name>/`):

   - `article_prompts_<folder_name>_final.csv`
     - Columns: `Article`, `Query`, `Response`
     - Contains all prompt–response pairs (for debugging/audit).

   - `article_results_<folder_name>_final.csv`
     - Columns: `Alloy name`, `Processing conditions`, `Characterization techniques`, `Properties`
     - Contains one row per alloy encountered across all relevant paragraphs.

   - `command_prompt_output_<folder_name>_final.txt`
     - Log of per‑DOI extraction: prompts, timings, counters, etc.

The main entry point is:

```python
from material_extractor import material_info_gpt_extractor

material_info_gpt_extractor("10.1016/j.msea.2008.08.025")
```

---

### 4.6. `run_batch.py`

Batch driver for processing many DOIs.

Reads the DOIs list from `DOIS_FILE` (default `./extracted_data_dois.txt`) using `read_dois_from_file`, then loops over a slice.

Example behavior:

- Log file: `GLOBAL_LOG_FILE` (default `./logs_dec_ver_q1_set.txt`)
- For each DOI:
  - Writes a start message to the global log
  - Calls `material_info_gpt_extractor(doi)`
  - Logs the time taken

CLI usage:

```bash
# Run on all DOIs in extracted_data_dois.txt
python run_batch.py

# Run from index 5904 to end
python run_batch.py 5904

# Run a specific slice [start_idx, end_idx)
python run_batch.py 5904 5905
```

To run in the background on a server:

```bash
nohup python3 run_batch.py 0 1000 > logs_dec_ver_q1_set.txt &
```

Adjust indices based on how many DOIs you want in that batch.

---

## 5. Data Layout

### 5.1. DOIs List

`extracted_data_dois.txt` (path configurable via `DOIS_FILE`) should contain one DOI per line, for example:

```text
10.1016/j.msea.2006.11.049
10.1016/j.jallcom.2008.11.059
10.1016/j.matdes.2012.08.019
```

### 5.2. Article Text

For each DOI, there should be a folder under `ARTICLES_ROOT`:

```text
ARTICLES_ROOT/
└── <folder_name>/
    └── Processed_Text_Data.csv
```

The `folder_name` is typically derived via `doi_to_foldername(doi)`.

`Processed_Text_Data.csv` must contain at least:

- `Section Heading`
- `Section Title`
- `Paragraph`

You can generate this file from PDF + OCR pipelines or separate preprocessing notebooks.

### 5.3. Example Embeddings

The example lists in `rag_examples.py` should be embedded and saved once to `.pkl` files in `EXAMPLES_EMBEDDINGS_DIR`:

```text
examples_embeddings/
├── alloy_embeddings.pkl
├── processing_condition_embeddings.pkl
├── characterization_embeddings.pkl
└── properties_embeddings.pkl
```

To regenerate them (only needed if you change `rag_examples.py`):

```python
from embeddings_utils import precompute_and_save_embeddings
from rag_examples import (
    alloy_examples,
    processing_condition_examples,
    characterization_examples,
    properties_examples,
)

precompute_and_save_embeddings(alloy_examples, "./examples_embeddings/alloy_embeddings.pkl")
precompute_and_save_embeddings(processing_condition_examples, "./examples_embeddings/processing_condition_embeddings.pkl")
precompute_and_save_embeddings(characterization_examples, "./examples_embeddings/characterization_embeddings.pkl")
precompute_and_save_embeddings(properties_examples, "./examples_embeddings/properties_embeddings.pkl")
```

---

## 6. Running an Example

1. Prepare your environment:

   ```bash
   cp .env.example .env
   # Edit .env and fill in your OpenAI keys and paths

   python -m venv .venv
   source .venv/bin/activate      # On Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. Ensure you have:

   - `extracted_data_dois.txt` populated
   - `articles_all/<folder_name>/Processed_Text_Data.csv` for each DOI
   - Precomputed `examples_embeddings/*.pkl`

3. Run extraction for a single DOI in an interactive Python shell:

   ```python
   from material_extractor import material_info_gpt_extractor

   material_info_gpt_extractor("doi")
   ```

4. Or run batch extraction over multiple DOIs:

   ```bash
   python run_batch.py 0 100
   ```

   This processes DOIs with indices `[0, 100)` from `extracted_data_dois.txt`.

---
## 7. Post-processing

The notebook `QuerySet1PostProcessingCode.ipynb` implements the post-processing pipeline used to construct the final database from the extracted records. Since identical alloy systems, defined by the same composition and processing conditions, may be discussed across multiple paragraphs, this step identifies and consolidates such entries into unified records. The notebook also contains the logic for normalizing alloy compositions to ensure consistency across the dataset. In addition, it reports summary statistics that provide an overview of the final database.

## 8. Extracted Data

The final database can be found in `QuerySet1-Database.csv`. The alloy compositions that occur in more than one paper can be found in `QuerySet1-repeated_alloys.csv`. The reference table used for evaluation can be found in `HEAs Table.docx` whose source can be found in the main paper.

## 9. Notes and Limitations

- This pipeline is tuned for **multicomponent / high-entropy alloy** literature.
- Quality depends on:
  - The **text preprocessing** (how `Processed_Text_Data.csv` is generated)
  - The quality and coverage of **few-shot examples**
  - The current behavior and version of the OpenAI models.
- Token usage is approximated via character length; if needed, you can integrate an exact tokenizer later.
- There is a heuristic (`found_special` indices in FAISS) to decide when to "upgrade" a query to the strong model (GPT‑4o) for alloy extraction.

---

## License

