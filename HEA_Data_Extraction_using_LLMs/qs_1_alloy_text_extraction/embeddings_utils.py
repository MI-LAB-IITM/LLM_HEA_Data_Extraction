# embeddings_utils.py

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

from openai_clients import get_embedding


def preprocess_text(text: str) -> np.ndarray:
    """
    Convert text into an embedding vector as a numpy array.
    """
    emb = get_embedding(text)
    return np.array(emb, dtype=np.float32)


def precompute_and_save_embeddings(example_texts: List[str], filename: str) -> np.ndarray:
    """
    Compute embeddings for a list of example texts and save them to a pickle file.
    """
    embeddings = np.array([preprocess_text(text) for text in example_texts], dtype=np.float32)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings


def load_embeddings(filename: str) -> np.ndarray:
    """
    Load precomputed embeddings from a pickle file.
    """
    with open(filename, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def find_similar_examples(
    example_embeddings: np.ndarray,
    user_paragraph: str,
    examples: List[str],
    k: int = 3,
) -> Tuple[bool, List[str]]:
    """
    Use FAISS to find top-k similar examples to the given user paragraph.

    Returns
    -------
    found_special : bool
        Whether any of the indices {24, 25} appear in the top-k (your original heuristic).
    top_examples : list[str]
        The k example strings corresponding to the nearest neighbors.
    """
    user_embedding = preprocess_text(user_paragraph)
    user_embedding_np = user_embedding.reshape(1, -1)

    # Build FAISS index
    d = example_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(example_embeddings)

    distances, indices = index.search(user_embedding_np, k)

    values_to_search = {24, 25}
    found_special = any(int(v) in values_to_search for v in indices[0])

    top_examples = [examples[int(idx)] for idx in indices[0]]

    return found_special, top_examples


def find_similar_examples_from_precomputed(
    user_paragraph: str,
    embedding_file: str,
    examples: List[str],
    k: int = 3,
) -> Tuple[bool, List[str]]:
    """
    Convenience wrapper: load embeddings from file and run similarity search.
    """
    example_embeddings = load_embeddings(embedding_file)
    return find_similar_examples(example_embeddings, user_paragraph, examples, k)
