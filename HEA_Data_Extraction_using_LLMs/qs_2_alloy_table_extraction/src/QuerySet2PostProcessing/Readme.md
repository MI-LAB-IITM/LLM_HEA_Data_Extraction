# Alloy Data Extraction & Validation Utilities

This repository contains modular utilities for extracting, normalizing, validating, categorizing, and retrieving alloy composition data from scientific literature, with support for LLM-assisted correction and FAISS-based retrieval.

---

## File Overview

### `acronym.py`
Parses alloy composition acronyms expressed as key–value pairs (e.g., `Fe:50;Ni:50`).  
- Standardizes compositions into sorted dictionaries
- Handles formatting noise (±, malformed values)
- Validates chemical correctness using the composition validator
- Serializes standardized compositions into a compact string form

---

### `alloy_finder.py`
Searches for alloy acronyms within processed article text.
- Normalizes Unicode text and dash variants
- Scans paragraph-level CSVs for exact acronym matches
- Tracks found, not-found, and frequency of matches per DOI

---

### `categorizer.py`
Classifies alloy entries as **Conventional** or **Unconventional**.
- Flags missing or malformed alloy names
- Handles element-only cases with numeric suffixes
- Uses formula parsing and composition validation for final classification

---

### `formula_parser.py`
Parses chemical formulas into element–fraction dictionaries.
- Supports single formulas or lists
- Uses regex-based element matching
- Rejects malformed or duplicate-element formulas

---

### `validator.py`
Validates chemical compositions.
- Checks numeric validity and bounds
- Verifies element symbols against the periodic table
- Expands compound formulas into elemental constituents
- Provides a single `valid_composition` interface

---

### `ollama_bot.py`
Wraps an Ollama LLM call to correct formatting errors.
- Sends system and user prompts
- Returns cleaned model output for downstream use

---

### `faissvs.py`
Implements a FAISS-based vector store for retrieval-augmented generation (RAG).
- Embeds alloy-related documents using OpenAI embeddings
- Stores vectors, metadata, and auxiliary fields
- Supports similarity search, context construction for LLMs, and persistence

---

## Intended Use
These modules together support an end-to-end pipeline for:
- Alloy composition extraction
- Validation and normalization
- Categorization and quality control
- Semantic retrieval and LLM-assisted reasoning

Each file is designed to be independently reusable but interoperable within a larger materials text-mining workflow.
