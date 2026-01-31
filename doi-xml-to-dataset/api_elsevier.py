from __future__ import annotations
import os, time, httpx, logging

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
import os

# Load .env file at the beginning
load_dotenv()

def fetch_xml_by_doi(
    doi: str,
    api_key: str | None = None,
    inst_token: str | None = None,
    timeout_s: int = 60,
    max_retries: int = 3,
    sleep_between_retries_s: int = 5,
) -> str | None:
    api_key = api_key or os.getenv("ELS_API_KEY")
    inst_token = inst_token or os.getenv("ELS_INST_TOKEN")
    if not api_key or not inst_token:
        raise RuntimeError("ELS_API_KEY and ELS_INST_TOKEN must be set")

    url = f"https://api.elsevier.com/content/article/doi/{doi}"
    headers = {
        "X-ELS-APIKey": api_key,
        "X-ELS-Insttoken": inst_token,
        "Accept": "text/xml",
    }
    timeout = httpx.Timeout(timeout_s, connect=timeout_s)

    with httpx.Client(timeout=timeout, headers=headers) as client:
        for attempt in range(1, max_retries + 1):
            try:
                r = client.get(url)
                if r.status_code == 200:
                    return r.text
                if r.status_code == 404:
                    logger.warning("DOI not found: %s", doi)
                    return None
                logger.warning("status=%s attempt %s/%s", r.status_code, attempt, max_retries)
            except httpx.TimeoutException as e:
                logger.warning("timeout %s attempt %s/%s", e, attempt, max_retries)
            time.sleep(sleep_between_retries_s)
    return None
