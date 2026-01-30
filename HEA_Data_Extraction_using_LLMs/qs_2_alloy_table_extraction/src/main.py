import os
import sys
import csv
import io
import time
import random
import logging
from io import StringIO
from typing import List, Set, Tuple

import faiss
import numpy as np
import pandas as pd
import openai


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.getLogger("httpx").setLevel(logging.WARNING)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please export it in your environment "
        "before running this script."
    )

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)


def exponential_backoff(retry_count: int) -> float:
    """Calculates the exponential backoff time."""
    return min(2 ** retry_count + random.random(), 60.0)


def generate_completion(model: str, prompt: str, max_retries: int = 5) -> str:
    """Wrapper to call chat.completions with streaming + exponential backoff."""
    for retry in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.0,
            )
            res = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    res += chunk.choices[0].delta.content
            return res
        except Exception as e:
            logging.warning(f"[{model}] Error: {e}")
            if retry == max_retries - 1:
                raise
            sleep_time = exponential_backoff(retry)
            logging.info(f"[{model}] Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)


def generate_completion_gpt_4(prompt: str) -> str:
    return generate_completion("gpt-4o", prompt)


def generate_completion_gpt_4_mini(prompt: str) -> str:
    return generate_completion("gpt-4o-mini", prompt)



material_properties_with_symbols = {
    "Yield Strength": ["σᵧ"],
    "Ultimate Tensile Strength": ["σₘₐₓ"],
    "Shear Strength": ["τ"],
    "Strain": ["ε"],
    "Stress": ["σ"],
    "Fatigue": ["Δσ"],
    "Creep": ["ε̇"],
    "Poisson's Ratio": ["ν"],
    "Damping Capacity": ["ζ"],
    "Thermal Expansion Coefficient": ["α"],
    "Thermal Conductivity": ["λ"],
    "Seebeck Coefficient": ["S", "α"],
    "Permittivity": ["ε"],
    "Charge Transfer Impedance": ["Z"],
    "Band Gap": ["E₉"],
    "Work Function": ["Φ"],
    "Dielectric Constant": ["κ", "εᵣ"],
    "Hall Coefficient": ["R_H"],
    "Electron Mean Free Path": ["λ"],
    "Magnetization": ["M"],
    "Saturation Magnetization": ["Mₛ"],
    "Coercivity": ["H_c"],
    "Curie Temperature": ["T_c"],
    "Magnetic Susceptibility": ["χ"],
    "Remanence": ["M_r"],
    "Magnetostriction": ["λ_s"],
    "Exchange Bias Field": ["H_ex"],
    "Magnetic Permeability": ["μ"],
    "Friction": ["μ"],
    "Core Loss": ["P_c"],
    "Elastic Moduli": ["E", "G", "B"],
    "Interfacial Energy": ["γ"],
    "Vacancy Formation Energy": ["E_v"],
    "Cohesive Energy": ["E_coh"],
    "Crack-Growth Rates": ["da/dN"],
    "Crystallographic Orientation": ["θ"],
    "Lattice Constants": ["a", "b", "c", "α", "β", "γ"],
    "Mean-Square Atomic Displacements": ["⟨u²⟩"],
    "Phase Transition Temperatures": ["T_t"],
    "Phase Transition": ["ΔH"],
    "Elasticity": ["E", "G", "B", "ν"],
    "Stress and Strain": ["σ", "ε"],
    "Electronegativity": ["χ"],
    "Zener Ratio": ["A"]
}

symbol_to_english = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    
    # Capital Greek letters
    "Α": "Alpha",
    "Β": "Beta",
    "Γ": "Gamma",
    "Δ": "Delta",
    "Ε": "Epsilon",
    "Ζ": "Zeta",
    "Η": "Eta",
    "Θ": "Theta",
    "Ι": "Iota",
    "Κ": "Kappa",
    "Λ": "Lambda",
    "Μ": "Mu",
    "Ν": "Nu",
    "Ξ": "Xi",
    "Ο": "Omicron",
    "Π": "Pi",
    "Ρ": "Rho",
    "Σ": "Sigma",
    "Τ": "Tau",
    "Υ": "Upsilon",
    "Φ": "Phi",
    "Χ": "Chi",
    "Ψ": "Psi",
    "Ω": "Omega",
    
    # Subscripts and special cases
    "σᵧ": "sigma_y",
    "σₘₐₓ": "sigma_max",
    "Δσ": "delta_sigma",
    "ε̇": "epsilon_dot",
    "εᵣ": "epsilon_r",
    "E₉": "E_g",
    "H_c": "H_c",
    "T_c": "T_c",
    "Mₛ": "M_s",
    "M_r": "M_r",
    "H_ex": "H_ex",
    "P_c": "P_c",
    "E_v": "E_v",
    "E_coh": "E_coh",
    "da/dN": "da_dN",
    "⟨u²⟩": "u_squared",
    "T_t": "T_t",
    "ΔH": "delta_H",
    
    # Multi-letter cases
    "a, b, c": "a_b_c",
    "α, β, γ": "alpha_beta_gamma",
    
    # Miscellaneous
    "S": "S",
    "Z": "Z",
    "R_H": "R_H",
    "M": "M",
    "G": "G",
    "B": "B",
    "A": "A",
    "E": "E"
}



def replace_symbols(text: str) -> str:
    for symbol, ascii_rep in symbol_to_english.items():
        text = text.replace(symbol, ascii_rep)
    return text


def normalize_query(query: str) -> str:
    """Convert non-ASCII characters in a query string to ASCII equivalents."""
    return replace_symbols(query)


class VectorStore:
    def __init__(
        self,
        embedding_dim: int = 1536,
        index_file: str = "vector_store.index",
        metadata_file: str = "metadata.pkl",
    ):
        self.embedding_dim = embedding_dim
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.text_data: List[str] = []

        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "rb") as f:
                import pickle

                self.text_data = pickle.load(f)
            logging.info("Loaded existing vector store.")
        except Exception:
            self.index = faiss.IndexFlatIP(embedding_dim)
            logging.info("Created new vector store.")

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate normalized embedding for a given text."""
        response = openai_client.embeddings.create(
            input=[text], model="text-embedding-3-large"
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding / (np.linalg.norm(embedding) + 1e-10)

    def add_texts(self, texts: List[str]) -> None:
        """Add texts to the vector store with normalization."""
        import pickle

        embeddings = np.array(
            [self.get_embedding(text) for text in texts], dtype=np.float32
        )
        self.index.add(embeddings)
        self.text_data.extend(texts)

        # Save index + metadata
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.text_data, f)

        logging.info("Vector store updated and saved.")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Search for top_k most similar texts using dot product."""
        query_embedding = self.get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        return [
            self.text_data[i] for i in indices[0] if 0 <= i < len(self.text_data)
        ]

    def print_all_entries(self) -> None:
        for i, text in enumerate(self.text_data):
            print(f"{i}: {text}\n")


store = VectorStore()

def process_csv_and_retrieve_properties(
    csv_string: str,
    store: VectorStore,
    similarity_threshold: float = 0.3,
    num_properties: int = 10,
) -> Set[str]:
    """
    Processes CSV data of candidate property strings, retrieves top properties
    from the vector store, and filters by similarity.
    """
    csv_reader = csv.reader(io.StringIO(csv_string))
    shortlisted: List[Tuple[str, float]] = []

    for row in csv_reader:
        for cell in row:
            if not isinstance(cell, str) or not cell.strip():
                continue

            normalized_query = normalize_query(cell.strip())
            try:
                query_embedding = store.get_embedding(normalized_query).reshape(1, -1)
            except Exception as e:
                logging.warning(
                    f"Skipping '{cell}' due to embedding error: {e}"
                )
                continue

            distances, indices = store.index.search(query_embedding, 20)

            query_top: List[Tuple[str, float]] = []
            seen_local = set()

            for idx, similarity in zip(indices[0], distances[0]):
                if idx < len(store.text_data) and similarity >= similarity_threshold:
                    property_name = store.text_data[idx].split(" [")[0]
                    if property_name not in seen_local:
                        query_top.append((property_name, similarity))
                        seen_local.add(property_name)

                    if len(query_top) == 3:
                        break

            shortlisted.extend(query_top)

    shortlisted.sort(key=lambda x: x[1], reverse=True)

    final_props: List[str] = []
    seen_global = set()

    for prop_name, _sim in shortlisted:
        if prop_name not in seen_global:
            final_props.append(prop_name)
            seen_global.add(prop_name)
        if len(final_props) == num_properties:
            break

    return set(final_props)


def create_prompt(table_caption: str, table_footer: str, table_data: str, properties: str) -> str:
    # Your original long instruction prompt here
    prompt = f"""Instructions:
1. The context section contains definitions of various material properties labelled by an expert...
...
Context(the property column in the extracted table strictly be these property names only):
{properties}

Table:
Table Caption: "{table_caption}"
Table Data:
{table_data}

Table Footer: "{table_footer if table_footer else ''}"
Question: Extract the properties of alloy systems present in the context from the following table data. Provide the data in comma-separated values (CSV) format only.
Output: """
    return prompt


def append_csv(csv_string: str, filename: str) -> None:
    lines = csv_string.strip().split("\n")
    header, data = lines[0], lines[1:]

    print("→ RAW CSV_STRING:\n", csv_string)
    print(f"→ Parsed header: {header!r}")
    print(f"→ Parsed {len(data)} data row(s)")
    for i, row in enumerate(data[:3]):
        print(f"   row {i+1}: {row!r}")

    file_exists = os.path.exists(filename)
    print(f"→ file_exists = {file_exists!r}, path = {filename!r}")

    if not file_exists:
        print(f"→ [append_csv] Creating {filename} and writing header + {len(data)} rows.")
        with open(filename, "a", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write("\n".join(data) + "\n")
    else:
        with open(filename, "r", encoding="utf-8") as existing_file:
            existing_header = existing_file.readline().strip()
        print(f"→ Existing file’s header: {existing_header!r}")
        if existing_header != header:
            raise ValueError(f"Header mismatch! {existing_header!r} != {header!r}")
        print(f"→ [append_csv] Appending {len(data)} rows to existing {filename}.")
        with open(filename, "a", encoding="utf-8") as f:
            f.write("\n".join(data) + "\n")


def process_table(
    table_path: str,
    table_caption: str,
    table_footer: str,
) -> Tuple[str, set, str, int, int, int, int]:
    try:
        df = pd.read_csv(table_path)
        df = df.astype(str)
        table_data = df.to_csv(index=False)

        df_temp = pd.read_csv(StringIO(table_data))
        r, c = df_temp.shape

        table_caption = str(table_caption)
        table_footer = str(table_footer)

        table = (
            f"Table caption: {normalize_query(table_caption)}\n"
            f"Table Data: {normalize_query(table_data)}\n"
            f"Table footer: {normalize_query(table_footer)}"
        )
        print(table)
        print("YES")

        # Step 1: detect property-like cells
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a materials science expert identifying cells in a table "
                        "that have material property names. The table caption and footer "
                        "too may contain property names. You must respond with cells "
                        "that you think contain a property name in a comma-separated "
                        "format. Make sure each of the properties in your response "
                        "mentioned are unique."
                    ),
                },
                {"role": "user", "content": table},
            ],
        )
        description = response.choices[0].message.content.strip()
        description = normalize_query(description)
        print(description)

        # Step 2: RAG over property descriptions
        retrieved_properties = process_csv_and_retrieve_properties(
            description, store, num_properties=10
        )
        print(retrieved_properties)

        # Step 3: load property descriptions
        df_properties = pd.read_csv("final_properties_with_descriptions_utf8.csv")
        filtered_df = df_properties[df_properties["Property"].isin(retrieved_properties)]
        filtered_df = filtered_df.reset_index(drop=True)

        properties_str = "\n\n".join(
            f"{i+1}. {row['Property']}: {row['Description']}"
            for i, row in filtered_df.iterrows()
        )

        prompt = create_prompt(
            normalize_query(table_caption),
            normalize_query(table_footer),
            normalize_query(table_data),
            normalize_query(properties_str),
        )

        gpt4_input_tokens = gpt4_output_tokens = 0
        gpt4_mini_input_tokens = gpt4_mini_output_tokens = 0

        if (r >= 6 or c >= 5) and r * c >= 25:
            gpt4_input_tokens = len(prompt)
            final_response = generate_completion_gpt_4(prompt)
            gpt4_output_tokens = len(final_response)
        else:
            gpt4_mini_input_tokens = len(prompt)
            final_response = generate_completion_gpt_4_mini(prompt)
            gpt4_mini_output_tokens = len(final_response)

        print(f"Input tokens: {gpt4_input_tokens}, Output tokens: {gpt4_output_tokens}")
        print(
            f"Input tokens (mini): {gpt4_mini_input_tokens}, "
            f"Output tokens (mini): {gpt4_mini_output_tokens}"
        )
        print(final_response)

        return (
            table_data,
            retrieved_properties,
            final_response,
            gpt4_input_tokens,
            gpt4_output_tokens,
            gpt4_mini_input_tokens,
            gpt4_mini_output_tokens,
        )

    except Exception as e:
        logging.error(f"Error processing table {table_path}: {e}")
        return None, None, None, 0, 0, 0, 0


def process_doi_folder(
    doi_folder: str,
    folder: str,
    final_output_file: str,
) -> bool:
    tables_folder = os.path.join(doi_folder, "tables")
    metadata_path = os.path.join(doi_folder, "Table_Metadata.csv")
    doi_output_folder = "query_set2_dois_processing_log_results"
    os.makedirs(doi_output_folder, exist_ok=True)

    final_output_folder = os.path.join(doi_output_folder, folder)
    os.makedirs(final_output_folder, exist_ok=True)

    log_path = os.path.join(final_output_folder, "processing_log.txt")
    success_found = False

    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file missing for {doi_folder}")
        return False

    metadata_df = pd.read_csv(metadata_path)
    MULTI_ATTEMPTS_LOG = "multi_attempts_log.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:
        for _, row in metadata_df.iterrows():
            table_label = row["Table Label"]
            table_caption = row["Table Caption"]
            table_footer = row["Table Footer"]
            table_path = os.path.join(tables_folder, f"{table_label}.csv")

            if not os.path.exists(table_path):
                logging.warning(f"Table file {table_path} missing.")
                log_file.write(f"Table {table_label}: file missing.\n")
                continue

            max_attempts = 2
            final_output = None

            for attempt in range(1, max_attempts + 1):
                start_time = time.time()

                (
                    table_data,
                    retrieved_properties,
                    final_output,
                    gpt4_input_tokens,
                    gpt4_output_tokens,
                    gpt4_mini_input_tokens,
                    gpt4_mini_output_tokens,
                ) = process_table(table_path, table_caption, table_footer)

                duration = time.time() - start_time
                print(f"Processed {table_label} in {duration:.2f}s")

                if (
                    table_data is None
                    or retrieved_properties is None
                    or final_output is None
                ):
                    log_file.write(f"Table {table_label}: Error processing table.\n")
                    break

                log_file.write(f"Input Table: {table_data}\n")
                log_file.write(
                    "Retrieved Properties: "
                    f"{', '.join(retrieved_properties)}\n"
                )
                log_file.write(f"Final Output: {final_output}\n")
                log_file.write(f"Processing time: {duration:.2f}s\n")
                log_file.write(
                    f"GPT-4 Input Tokens: {gpt4_input_tokens}, "
                    f"Output Tokens: {gpt4_output_tokens}\n"
                )
                log_file.write(
                    f"GPT-4 Mini Input Tokens: {gpt4_mini_input_tokens}, "
                    f"Output Tokens: {gpt4_mini_output_tokens}\n"
                )

                lines = final_output.strip().split("\n")
                header, *data_lines = lines
                valid = all(len(r.split(",")) == 6 for r in data_lines)

                if valid:
                    break
                elif attempt < max_attempts:
                    log_file.write(
                        f"{table_label}: column-count mismatch, retrying…\n"
                    )
                    with open(MULTI_ATTEMPTS_LOG, "a", encoding="utf-8") as mlog:
                        mlog.write(f"{doi_folder}\n")
                else:
                    log_file.write(
                        f"{table_label}: Failed to produce 6 columns after "
                        f"{max_attempts} attempts\n"
                    )

            if final_output and final_output.strip() != "-":
                log_file.write(
                    f"Table {table_label}: ✔ Success, appending to CSV.\n"
                )
                print(f"→ writing to {final_output_file}")
                append_csv(final_output, final_output_file)
                success_found = True
            else:
                log_file.write(
                    f"Table {table_label}: ✘ No properties found "
                    f"(output was '{final_output}').\n"
                )
                print(f"Table {table_label}: No properties found in the table.\n")

    return success_found


def read_dois_from_file(input_file: str) -> List[str]:
    with open(input_file, "r", encoding="utf-8") as f:
        return [doi.strip() for doi in f]


def doi_to_foldername(doi: str) -> str:
    return doi.replace("./", "")


def process_dois(
    dois: List[str],
    root_folder: str,
    final_output_file: str,
    start_index: int,
) -> None:
    logging.info(f"Extraction batch: starting index {start_index}")
    with open("successful_dois_log.txt", "a", encoding="utf-8") as success_log:
        for offset, doi in enumerate(dois):
            k = start_index + offset
            logging.info(f"{k}: START DOI {doi}")
            try:
                folder = doi_to_foldername(doi)
                doi_folder = os.path.join(root_folder, folder)
                if not os.path.isdir(doi_folder):
                    raise FileNotFoundError(f"Folder not found: {doi_folder}")

                success = process_doi_folder(doi_folder, folder, final_output_file)
                if success:
                    success_log.write(f"{k}\t{doi}\n")
                logging.info(f"{k}: COMPLETED DOI {doi} (success={success})")
            except Exception as e:
                logging.error(f"{k}: FAILED DOI {doi}: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-based table property extraction for HEA articles."
    )
    parser.add_argument(
        "--root-folder",
        default="articles_all",
        help="Root folder containing per-DOI subfolders.",
    )
    parser.add_argument(
        "--input-file",
        default="extracted_data_dois.txt",
        help="File containing DOIs (one per line).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=9373,
        help="Start index into DOI list.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=11000,
        help="End index (Python slicing, exclusive).",
    )
    parser.add_argument(
        "--output-csv",
        default="final_outputs_table2_rag_v5.csv",
        help="Output CSV file to write extracted rows into.",
    )

    args = parser.parse_args()

    try:
        all_dois = read_dois_from_file(args.input_file)
        batch = all_dois[args.start_index : args.end_index]
        print(
            f"Processing DOIs[{args.start_index}:{args.end_index}] "
            f"-> {len(batch)} entries"
        )
        process_dois(batch, args.root_folder, args.output_csv, args.start_index)
    except Exception as e:
        logging.error(f"Fatal error, terminating: {e}", exc_info=True)
    finally:
        logging.info("Script terminated.")
