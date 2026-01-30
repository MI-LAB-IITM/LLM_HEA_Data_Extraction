# openai_clients.py

import os
import random
import time
from typing import Literal

from openai import OpenAI

# --------- Configuration ---------

# Prefer specific keys, fall back to OPENAI_API_KEY
_EMBED_KEY = os.getenv("OPENAI_API_KEY_EMBED") or os.getenv("OPENAI_API_KEY")
_GPT4O_KEY = os.getenv("OPENAI_API_KEY_GPT4O") or os.getenv("OPENAI_API_KEY")
_GPT4O_MINI_KEY = os.getenv("OPENAI_API_KEY_GPT4O_MINI") or os.getenv("OPENAI_API_KEY")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL_STRONG = os.getenv("CHAT_MODEL_STRONG", "gpt-4o")
CHAT_MODEL_LIGHT = os.getenv("CHAT_MODEL_LIGHT", "gpt-4o-mini")

# --------- Clients ---------

client_embed = OpenAI(api_key=_EMBED_KEY)
client_strong = OpenAI(api_key=_GPT4O_KEY)
client_light = OpenAI(api_key=_GPT4O_MINI_KEY)


# --------- Embeddings ---------

def get_embedding(text: str, model: str = EMBED_MODEL) -> list[float]:
    """
    Get an embedding vector for the given text using OpenAI's embedding API.
    Newlines are replaced with spaces for stability.
    """
    text = text.replace("\n", " ")
    response = client_embed.embeddings.create(
        input=[text],
        model=model,
    )
    return response.data[0].embedding


# --------- Chat Completions with Exponential Backoff ---------

def _exponential_backoff_delay(retry_count: int) -> float:
    """
    Calculates exponential backoff delay with jitter, capped at 60 seconds.
    """
    return min(2 ** retry_count + random.random(), 60.0)


def generate_chat_completion(
    prompt: str,
    mode: Literal["strong", "light"] = "light",
    max_retries: int = 5,
    temperature: float = 0.0,
    stream: bool = True,
) -> str:
    """
    Generate a completion from OpenAI chat models with retries and exponential backoff.

    Parameters
    ----------
    prompt : str
        The user prompt.
    mode : {"strong", "light"}
        "strong" uses CHAT_MODEL_STRONG (e.g. GPT-4o),
        "light" uses CHAT_MODEL_LIGHT (e.g. GPT-4o-mini).
    max_retries : int
        Maximum number of retries on failure.
    temperature : float
        Sampling temperature.
    stream : bool
        If True, use streaming API and accumulate content.

    Returns
    -------
    str
        The full text response from the model.
    """
    if mode == "strong":
        client = client_strong
        model_name = CHAT_MODEL_STRONG
    else:
        client = client_light
        model_name = CHAT_MODEL_LIGHT

    for retry in range(max_retries):
        try:
            if stream:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    temperature=temperature,
                )
                out = []
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        out.append(delta.content)
                return "".join(out)
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[OpenAI error] {e}")
            if retry == max_retries - 1:
                raise
            delay = _exponential_backoff_delay(retry)
            print(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
