"""
Shared utilities: structured logging and a resilient Ollama call wrapper.

Centralizing these means every module logs consistently and every Ollama call
gets the same retry/backoff behavior, so a transient connection drop during a
multi-hour eval doesn't kill the whole run.
"""
import logging
import sys
import time

import ollama


def get_logger(name: str) -> logging.Logger:
    """Return a module logger configured once with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


_log = get_logger("ollama_call")


def ollama_chat(model: str, prompt: str, options: dict | None = None,
                retries: int = 3, backoff: float = 2.0) -> str:
    """
    Call Ollama's chat endpoint with retries and exponential backoff.

    Returns the assistant message content as a string. Raises the last exception
    if every attempt fails, so the caller can decide how to handle a hard failure.
    """
    options = options or {}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
            )
            return resp["message"]["content"]
        except Exception as e:  # noqa: BLE001 - we genuinely want to retry on anything transient
            last_err = e
            if attempt < retries:
                wait = backoff ** attempt
                _log.warning("Ollama call failed (attempt %d/%d): %s — retrying in %.1fs",
                             attempt, retries, e, wait)
                time.sleep(wait)
            else:
                _log.error("Ollama call failed after %d attempts: %s", retries, e)
    raise last_err