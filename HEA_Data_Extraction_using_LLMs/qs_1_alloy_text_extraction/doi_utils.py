# doi_utils.py

from typing import List


def save_dois_to_file(dois_list: List[str], output_file: str) -> None:
    """
    Save a list of DOIs to a text file, one DOI per line.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for doi in dois_list:
            f.write(doi.strip() + "\n")


def read_dois_from_file(input_file: str) -> List[str]:
    """
    Read DOIs from a text file, stripping whitespace and newlines.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def doi_to_foldername(doi: str) -> str:
    """
    Convert a DOI string to a folder-friendly name.

    Your original code used `replace("./", "")`, but typically DOIs don't
    contain "./". Keep that logic if your input includes such prefixes.
    """
    return doi.replace("./", "")
