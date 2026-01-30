# material_extractor.py

import csv
import os
import sys
import time
from typing import List, Set

import pandas as pd

from doi_utils import doi_to_foldername
from embeddings_utils import find_similar_examples_from_precomputed
from openai_clients import generate_chat_completion
from rag_examples import (
    alloy_examples,
    processing_condition_examples,
    characterization_examples,
    properties_examples,
)

# --------- Paths from environment or defaults ---------

ARTICLES_ROOT = os.getenv("ARTICLES_ROOT", "./articles_all")
RESULTS_ROOT = os.getenv("RESULTS_ROOT", "./query_set1_dois_results_dec_ver_all")
EXAMPLES_EMBEDDINGS_DIR = os.getenv("EXAMPLES_EMBEDDINGS_DIR", "./examples_embeddings")

ALLOY_EMB_FILE = os.path.join(EXAMPLES_EMBEDDINGS_DIR, "alloy_embeddings.pkl")
PROC_EMB_FILE = os.path.join(EXAMPLES_EMBEDDINGS_DIR, "processing_condition_embeddings.pkl")
CHAR_EMB_FILE = os.path.join(EXAMPLES_EMBEDDINGS_DIR, "characterization_embeddings.pkl")
PROP_EMB_FILE = os.path.join(EXAMPLES_EMBEDDINGS_DIR, "properties_embeddings.pkl")


def _build_alloy_prompt(article_text: str, examples_for_alloy: List[str], query: str) -> str:
    """
    Build the long alloy extraction prompt with instructions and examples.
    """
    prompt_examples = "\n".join([f"{i+1}. {ex}" for i, ex in enumerate(examples_for_alloy)])

    prompt = f"""
You are a materials engineer.
-Keep the response to just one word.
-If you don't have any context and are unsure of the answer, reply with '-' a single character only as your response.
-If more than one answer is present, name all of them separated by comma.
-Do not use the terms 'HEA' or 'HE' if a specific alloy is mentioned. If no specific alloy is found, reply with '-' character.
-If the alloy has x mentioned in it, substitute values of x mentioned within the article to get alloy composition. For substitutions like x, if the values are not given in the article, provide the composition with x itself.
-Strictly Do not provide alloy compositions as Al, Cu, Ni (single elements) or Cr-Al-Ni. Instead, specify the full composition.
-Strictly follow the three steps below

Step 1: Read the information below. Answer the user query regarding the information given in the article.
Universal information :
1. An alloy is a solid mixture of multiple elements. Its chemical composition conveys the amount of each element present in the alloy. For example, an alloy with chemical composition AlCoCr contains elements Al, Co and Cr in equal amount. Similarly, an alloy with chemical composition Al2Co3Ni5 contains 2 parts Al, 3 parts Co and 5 parts Ni.
2. Alloy compositions can be explicitly mentioned. For example, "An alloy system was synthesized through arc melting the constituents at 1500K to get Al2Mo4Cr6, AlMo3Cr5". The alloy compositions mentioned are Al2Mo4Cr6, AlMo3Cr5.
3. There are two other ways in which alloys can be represented, in terms of molar ratio or atomic ratio.
4. In terms of molar ratio, "Multicomponent AlCoCrCuFeNiMox(x values in molar ratio, x =0, 0.2, 0.8 and 1.0) alloys were prepared using a well-developed copper mould casting." 4 alloy compositions are mentioned AlCoCrCuFeNi,AlCoCrCuFeNiMo0.2, AlCoCrCuFeNiMo0.8,AlCoCrCuFeNiMo.
5. Below is a general convention by which an alloy with varying percentage of an element is reported. For example, AlxTiVCr(x=1, 2) refers to two alloys with composition AlTiVCr when x=1, and Al2TiVCr when x=2. However, in some cases alloy compositions are reported as atomic percentages, for example Alx(TiVCr)100-x(x=25, and 40 at%) refers to two alloys with compositions Al25Ti25V25Cr25 and Al40Ti20V20Cr20, where in the former case with x=25 Al amounts to 25% and TiVCr together contribute to 75 %, which when equally distributed among the three elements gives us Al25Ti25V25Cr25. Similarly for the latter case, Al amounts to 40% and TiVCr together amounts to 60%, which when equally divided results in Al40Ti20V20Cr20. In cases with brackets like this, we do the task in steps: first step evaluate the amount to be split, so when x=25, 100-x is 75%. Second, count the number of elements that we split it among, TiVCr in the brackets are 3 elements, so 75% is split into 3 so 75/3=25% which gives us Al25Ti25V25Cr25. Follow the 2 steps and do the calculation accurately.

Step 2: Go through these example prompts for extracting alloy composition:
{prompt_examples}

Step 3: Answer the user question. Make sure that the answer is from the article. Take your time with the task, accuracy is important:

User Question: {query}
Article: {article_text}
"""
    return prompt


def _build_subprompt(
    article_text: str,
    alloy: str,
    examples_3: List[str],
    mode: str,
) -> str:
    """
    Build prompts for processing conditions, characterization techniques, or properties.

    mode in {"processing", "characterization", "properties"}
    """
    prompt_examples = "\n".join([f"{i+1}. {ex}" for i, ex in enumerate(examples_3)])

    if mode == "processing":
        query = f"Based on the examples and context, extract the processing conditions of the alloy {alloy} mentioned in the article."
        instructions = """You are a materials engineer.
-Keep the response to just one word.
-If you don't have any context and are unsure of the answer, reply with '-' single character only as your response.
-If more than one answer is present, name all of them separated by comma.
-Strictly follow the steps below
Step 1: Read the information below. Answer the user query regarding the information given in the article.
Information:
-Some of the common techniques used for processing or synthesis and subsequent thermo-mechanical treatments of alloys include As-Cast which is also called arc melting; Additive Manufacturing; Bridgman solidification; Cold-Pressed; Mechanically Alloyed plus Spark Plasma Sintered; Melt-Spun; Sputter deposition; Splat Quenched.
-Abbreviations for subsequent thermo-mechanical processes are: Cold Rolled; Furnace Cooled; Forged; Hot Isostatic Pressing; High Pressure Torsion; Hot Rolled; Vacuum Hot Press; Water Quenched
"""
    elif mode == "characterization":
        query = f"Based on the examples and context, extract the characterization techniques used for the alloy {alloy} mentioned in the article."
        instructions = """You are a materials engineer.
-Keep the response to just one word.
-If you don't have any specific characterization techniques present or unsure of the answer, reply with '-' single character only as your response.
-If more than one answer is present, name all of them separated by comma.
-Strictly follow the three steps below
Step 1: Read the information below. Answer the user query regarding the information given in the article.
Information:
Characterization techniques in materials science analyze materials' structure, composition, and properties. These techniques provide crucial insights for material development and application.
"""
    else:  # properties
        query = f"Based on the examples and context, extract the properties of the alloy {alloy} mentioned in the article."
        instructions = """You are a materials engineer.
-Keep the response to just one word.
-If you don't have any context and are unsure of the answer, reply with '-' single character only as your response.
-If more than one answer is present, name all of them separated by comma.
-Strictly follow the steps below
"""

    prompt = f"""{instructions}

Step 2: Go through these example prompts:
{prompt_examples}

Step 3: Answer the user question. Make sure that the answer is from the article. Take your time with the task, accuracy is important:
User Question: {query}
Article: {article_text}
"""
    return prompt, query


def material_info_gpt_extractor(doi: str) -> None:
    """
    Run Query Set 1 extraction for a single DOI:
    - Read Processed_Text_Data.csv
    - Extract alloy names
    - For each alloy, extract processing conditions, characterization techniques, and properties
    - Save per-DOI prompts and results into RESULTS_ROOT/<folder_name>/

    This function prints progress and timing info and is designed to be
    called from run_batch.py.
    """
    header = ["Alloy name", "Processing conditions", "Characterization techniques", "Properties"]
    query_list = [
        "Follow the examples and universal data and use it to extract the alloy compositions mentioned in the article."
    ]

    folder_name = doi_to_foldername(doi)
    folder_path_articles = os.path.join(ARTICLES_ROOT, folder_name)

    print(f"Input folder: {folder_path_articles}")
    if not os.path.exists(folder_path_articles):
        print(f"Folder for DOI {doi} does not exist. Skipping...")
        return

    folder_path_results = os.path.join(RESULTS_ROOT, folder_name)

    if os.path.exists(folder_path_results):
        print(f"Folder for DOI {doi} already extracted. Skipping...")
        return
    else:
        os.makedirs(folder_path_results, exist_ok=True)

    file_read = os.path.join(folder_path_articles, "Processed_Text_Data.csv")
    filename_prompts = os.path.join(folder_path_results, f"article_prompts_{folder_name}_final.csv")
    filename_results = os.path.join(folder_path_results, f"article_results_{folder_name}_final.csv")
    logfile_path = os.path.join(folder_path_results, f"command_prompt_output_{folder_name}_final.txt")

    total_prompts = 0
    total_input_chars_light = 0
    total_output_chars_light = 0
    total_input_chars_strong = 0
    total_output_chars_strong = 0

    # Redirect stdout into per-DOI log file
    log_handle = open(logfile_path, "w", encoding="utf-8")
    sys.stdout = log_handle

    start = time.time()

    article_prompts: List[List[str]] = []
    article_results: List[List[str]] = []

    # ---- Read / filter CSV ----
    df = pd.read_csv(file_read, encoding="utf-8")
    filtered_df = df[df["Section Heading"].astype(str).str.startswith(("0", "2"))]

    # ---- Step 1: Extract all alloy compositions for this article ----
    all_alloys: Set[str] = set()

    for _, row in filtered_df.iterrows():
        section_title = str(row["Section Title"])
        paragraph = str(row["Paragraph"])
        article = f"{section_title}: {paragraph}"

        flag, examples_for_alloy = find_similar_examples_from_precomputed(
            article, ALLOY_EMB_FILE, alloy_examples, k=4
        )

        for query_ in query_list:
            prompt = _build_alloy_prompt(article, examples_for_alloy, query_)
            print(f"prompt (alloy): {prompt}")
            print(f"prompt_length: {len(prompt)}")

            prompt_start = time.time()
            mode = "strong" if flag else "light"
            response = generate_chat_completion(prompt, mode=mode, temperature=0.0, stream=True)
            prompt_end = time.time()
            duration = prompt_end - prompt_start

            print(f"Time taken for alloy extraction from prompt: {duration:.2f} seconds")
            print("response:", response)

            alloys = [x.strip() for x in response.split(",")]
            print("parsed alloys:", alloys)

            article_prompts.append([article, query_, response])

            total_prompts += 1
            if mode == "strong":
                total_input_chars_strong += len(prompt)
                total_output_chars_strong += len(response)
            else:
                total_input_chars_light += len(prompt)
                total_output_chars_light += len(response)

            for alloy in alloys:
                if alloy and alloy != "-":
                    all_alloys.add(alloy)

    # ---- Step 2: For each alloy, extract processing, characterization, properties ----
    for alloy in all_alloys:
        for _, row in filtered_df.iterrows():
            section_title = str(row["Section Title"])
            paragraph = str(row["Paragraph"])
            article = f"{section_title}: {paragraph}"

            row_result = [alloy]

            # c = 0: processing, c = 1: characterization, c = 2: properties
            for c in range(3):
                if c == 0:
                    examples_3 = find_similar_examples_from_precomputed(
                        article, PROC_EMB_FILE, processing_condition_examples, k=4
                    )[1]
                    mode_name = "processing"
                elif c == 1:
                    examples_3 = find_similar_examples_from_precomputed(
                        article, CHAR_EMB_FILE, characterization_examples, k=4
                    )[1]
                    mode_name = "characterization"
                else:
                    examples_3 = find_similar_examples_from_precomputed(
                        article, PROP_EMB_FILE, properties_examples, k=4
                    )[1]
                    mode_name = "properties"

                prompt, query = _build_subprompt(article, alloy, examples_3, mode_name)
                print(f"prompt ({mode_name}): {prompt}")
                print(f"prompt_length: {len(prompt)}")

                response = generate_chat_completion(prompt, mode="light", temperature=0.0, stream=True)
                print("response:", response)

                total_prompts += 1
                total_input_chars_light += len(prompt)
                total_output_chars_light += len(response)

                extracted_final = "".join([ch if ord(ch) < 128 else " " for ch in response])
                row_result.append(extracted_final)

                if c == 0:
                    article_prompts.append([article, query, response])

            article_results.append(row_result)

    end = time.time()
    time_duration = end - start

    print(f"\n\nThe total time taken to extract the values: {time_duration:.2f} seconds")
    print(f"Total number of prompts used: {total_prompts}")
    print(f"Total input chars (light): {total_input_chars_light}")
    print(f"Total output chars (light): {total_output_chars_light}")
    print(f"Total input chars (strong): {total_input_chars_strong}")
    print(f"Total output chars (strong): {total_output_chars_strong}")

    # ---- Save CSVs ----
    with open(filename_prompts, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Article", "Query", "Response"])
        writer.writerows(article_prompts)

    with open(filename_results, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(article_results)

    print(f"\nData for DOI {doi} has been processed and saved in {folder_path_results}\n")

    # restore stdout
    log_handle.close()
    sys.stdout = sys.__stdout__
