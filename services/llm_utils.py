"""Shared LLM utilities — JSON cleanup, safe parsing, and LLM factory.

Eliminates duplicated JSON-fence-stripping code across cv_agent,
hr_agent, and email_agent.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def clean_llm_json(content: str) -> str:
    """Strip markdown code fences and whitespace from LLM output.

    Handles outputs wrapped in ```json ... ```, ``` ... ```,
    or plain JSON strings.
    """
    content = content.strip()

    # Remove leading ```json or ``` line
    if content.startswith("```"):
        # Remove first line (```json or ```)
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]

    # Remove trailing ```
    if content.endswith("```"):
        content = content[:-3]

    return content.strip()


def safe_parse_json(content: str, model: type[T]) -> T:
    """Clean LLM output and parse it into a Pydantic model.

    Args:
        content: Raw LLM output (may contain markdown fences).
        model: Pydantic model class to validate against.

    Returns:
        Validated Pydantic model instance.

    Raises:
        json.JSONDecodeError: If the content is not valid JSON.
        ValidationError: If the JSON does not match the model schema.
    """
    cleaned = clean_llm_json(content)
    data = json.loads(cleaned)
    return model(**data) if not isinstance(data, list) else data


def safe_parse_json_list(content: str) -> list[dict]:
    """Clean LLM output and parse it as a JSON array.

    Args:
        content: Raw LLM output expected to be a JSON array.

    Returns:
        List of dictionaries.

    Raises:
        json.JSONDecodeError: If the content is not valid JSON.
    """
    cleaned = clean_llm_json(content)
    data = json.loads(cleaned)
    if isinstance(data, list):
        return data
    return [data]


def create_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Create a ChatOpenAI instance with the configured model and API key.

    Centralizes LLM creation to ensure consistent configuration.
    """
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )
