# run_batch.py

import os
import sys
import time
from typing import Optional

from doi_utils import read_dois_from_file
from material_extractor import material_info_gpt_extractor

DOIS_FILE = os.getenv("DOIS_FILE", "./extracted_data_dois.txt")
GLOBAL_LOG_FILE = os.getenv("GLOBAL_LOG_FILE", "./logs_dec_ver_q1_set.txt")


def run_batch(start_idx: int = 0, end_idx: Optional[int] = None) -> None:
    """
    Run material_info_gpt_extractor on a slice of DOIs list.

    Parameters
    ----------
    start_idx : int
        Starting index in the DOIs list (inclusive).
    end_idx : Optional[int]
        Ending index (exclusive). If None, runs to the end of the list.
    """
    dois = read_dois_from_file(DOIS_FILE)

    if end_idx is None or end_idx > len(dois):
        end_idx = len(dois)

    os.makedirs(os.path.dirname(GLOBAL_LOG_FILE) or ".", exist_ok=True)
    global_log_handle = open(GLOBAL_LOG_FILE, "a", encoding="utf-8")

    start_time = time.time()
    print(f"[GLOBAL] Extraction started from index {start_idx} to {end_idx - 1}...")
    global_log_handle.write(f"Extraction started from {start_idx} to {end_idx - 1}\n")
    global_log_handle.flush()

    k = start_idx
    for doi in dois[start_idx:end_idx]:
        t0 = time.time()
        print(f"[GLOBAL] Starting DOI index {k}: {doi}")
        global_log_handle.write(f"Starting DOI index {k}: {doi}\n")
        global_log_handle.flush()

        material_info_gpt_extractor(doi)

        t1 = time.time()
        duration = t1 - t0
        msg = f"{k} Extraction {doi} done in {duration:.2f} seconds!\n"
        print("[GLOBAL]", msg.strip())
        global_log_handle.write(msg)
        global_log_handle.flush()
        k += 1

    total_duration = time.time() - start_time
    global_log_handle.write(
        f"\nThe total time taken to extract the values: {total_duration:.2f} seconds\n"
    )
    global_log_handle.close()


if __name__ == "__main__":
    # Simple CLI: python run_batch.py [start_idx] [end_idx]
    # Example:
    #   python run_batch.py          # all DOIs
    #   python run_batch.py 1 1500
    if len(sys.argv) == 1:
        run_batch()
    elif len(sys.argv) == 2:
        run_batch(int(sys.argv[1]))
    else:
        run_batch(int(sys.argv[1]), int(sys.argv[2]))
