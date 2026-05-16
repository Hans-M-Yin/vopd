#!/usr/bin/env python3
"""Rewrite the first prompt element in a parquet dataset for Qwen3-style prompting."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any
import io
import pandas as pd


NEW_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "You FIRST think about the reasoning process as an internal monologue and then "
    "provide the final answer. Close your answer with one or few words in \\boxed{}.\n"
)


def _update_first_prompt_item(prompt: Any) -> Any:
    """Replace the first prompt item with the target system prompt."""
    # print(prompt, type(prompt))
    # if not isinstance(prompt, list):
    #     raise TypeError(f"Expected 'prompt' to be a list, but got {type(prompt).__name__}.")
    # if not prompt:
    #     raise ValueError("Expected 'prompt' to contain at least one element.")
    updated_prompt = deepcopy(prompt)
    first_item = updated_prompt[0]

    if isinstance(first_item, dict):
        updated_first = dict(first_item)
        updated_first["content"] = NEW_SYSTEM_PROMPT
        if "role" in updated_first:
            updated_first["role"] = "system"
        updated_prompt[0] = updated_first
        return updated_prompt

    if isinstance(first_item, str):
        updated_prompt[0] = NEW_SYSTEM_PROMPT
        return updated_prompt

    raise TypeError(
        "Expected the first element of 'prompt' to be either a string or a dict, "
        f"but got {type(first_item).__name__}."
    )


def rewrite_prompt_column(input_path: Path, output_path: Path, prompt_column: str) -> None:
    """Load a parquet file, rewrite prompt[0], and save the result."""
    dataframe = pd.read_parquet(input_path, dtype_backend="pyarrow")

    if prompt_column not in dataframe.columns:
        raise KeyError(
            f"Column '{prompt_column}' not found in {input_path}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe[prompt_column] = dataframe[prompt_column].apply(_update_first_prompt_item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    dataframe.to_parquet(buffer, engine="pyarrow", index=False)

    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace the first element of each prompt in a parquet dataset with a Qwen3 system prompt."
    )
    parser.add_argument("input_parquet", type=Path, help="Input parquet file.")
    parser.add_argument("output_parquet", type=Path, help="Output parquet file.")
    parser.add_argument(
        "--prompt-column",
        default="prompt",
        help="Name of the column that stores the prompt list. Defaults to 'prompt'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rewrite_prompt_column(args.input_parquet, args.output_parquet, args.prompt_column)


if __name__ == "__main__":
    main()
