import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from llm_sdk import Small_LLM_Model as Small_LLM_ModelBase

DEFAULT_FUNCTIONS_DEFS_PATH = Path("input/functions_definition.json")
DEFAULT_TESTS_PATH = Path("input/function_calling_tests.json")
DEFAULT_OUTPUT_PATH = Path("output/function_calling_results.json")


def load_json_file(file_path: Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def build_tools_description(functions_list: list[dict[Any, Any]]) -> str:
    tools_info = ""
    for fn in functions_list:
        tools_info += f"Tool Name: {fn['name']}\n"
        tools_info += f"Description: {fn['description']}\n"
        tools_info += f"Parameters: {fn['parameters']}\n"
        tools_info += "\n"

    return f"""
You are a tool-calling assistant.

If the user request can be solved with a tool,
respond ONLY with valid JSON.

The JSON should have the following format:
Example:
Every detail:
{tools_info}
Vowels:[aeiouAEIOU]
Do not add text before or after the JSON.
It must end with the closing curly brace of the JSON.
"""


class CallMeMaybe(Small_LLM_ModelBase):  # type: ignore[misc]
    def __init__(
        self,
        definitions_path: str | Path = DEFAULT_FUNCTIONS_DEFS_PATH,
        tests_path: str | Path = DEFAULT_TESTS_PATH,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
    ):
        super().__init__()
        self.definitions_path = Path(definitions_path)
        self.tests_path = Path(tests_path)
        self.output_path = Path(output_path)

        self.functions = load_json_file(self.definitions_path)
        self.tools_description = build_tools_description(self.functions)

        self.id_json_open = self.encode("{")[0].tolist()[0]
        self.id_json_close = self.encode("}")[0].tolist()[0]
        self.id_quote = self.encode('"')[0].tolist()[0]
        self.id_colon = self.encode(":")[0].tolist()[0]

        # Pre-encode fixed tokens to simply extend the input_ids without
        # running inference
        self.name_token = self.encode(' "name": "')[0].tolist()
        self.params_open_token = self.encode(', "parameters": {')[0].tolist()

    def get_func_definitions(self) -> Any:
        # Return the cached functions loaded during __init__
        return self.functions

    def generate(self, input_ids: list[Any], user_request: str) -> Any:
        prompt_len = len(input_ids)

        # Correctly escape internal quotes in the prompt
        safe_user_request = json.dumps(user_request, ensure_ascii=False)
        request = self.encode(f'"prompt": {safe_user_request},')[0].tolist()

        # Fixed pre-encoded tokens that we will append using extend
        name = self.name_token
        param = self.params_open_token

        # --- CONSTRAINED DECODING ---
        # Force the first generated token to be the opening brace '{'
        input_ids.extend([self.id_json_open])
        # ------------------------------------------------------

        # Append the encoded prompt request
        input_ids.extend(request)
        # Append the pre-encoded 'name' key
        input_ids.extend(name)

        # Loop until the generated text matches one of the available
        # function names
        # r: string to decode the recent tokens and check what has
        # been generated
        r = ""
        token_index = 0
        funcs = self.get_func_definitions()
        func_name = {}
        for f in funcs:
            func_name[f["name"]] = f["description"], f["parameters"]
        # Handle the case where the model needs to pick an exact function name
        # default is 50 tokens
        # but we can break early if we match a function name
        for _ in range(50):
            orig_logits = self.get_logits_from_input_ids(input_ids)
            constrained_logits = np.full_like(orig_logits, -float("inf"))
            any_candidate = False

            # Build constraints based on the next expected token
            # for candidate function names
            for func in func_name.keys():
                token_ids = self.encode(func)[0].tolist()
                if func.startswith(r):
                    # If there's a next token at position token_index,
                    # allow it, otherwise we have already matched the full
                    # function string and should stop.
                    if token_index < len(token_ids):
                        func_encoded = token_ids[token_index]
                        constrained_logits[func_encoded] = (
                            orig_logits[func_encoded])
                    any_candidate = True

            # If no candidate tokens were set, fall back to original logits
            if not any_candidate or np.isneginf(constrained_logits).all():
                logits = orig_logits
            else:
                logits = constrained_logits
            next_token_id = int(np.argmax(logits))
            token_str = self.decode([next_token_id])
            r += token_str
            input_ids.append(next_token_id)
            token_index += 1

            # If we've matched a full function name,
            # stop generating further tokens
            if r in func_name:
                break
        if r not in func_name:
            # If we exit the loop without a valid function name
            # raise value error
            raise ValueError("Failed to generate a valid function name")
        token_index = 0
        a = self.encode('"')[0].tolist()
        input_ids.extend(a)
        input_ids.extend(param)

        def get_function(
            name: str,
            functions: list[Any]
        ) -> dict[Any, Any] | None:
            return next((fn for fn in functions if fn["name"] == name), None)

        fn = get_function(r, self.functions)
        params = fn["parameters"] if fn else {}

        def escape_json_string_fragment(fragment: str) -> str:
            # Keep generated string content JSON-safe while building output
            return (
                fragment
                .replace('\\', '\\')
                .replace('\n', '\n')
                .replace('\r', '\r')
                .replace('\t', '\t')
            )

        # Iteratively process each parameter
        for p, t in params.items():
            key = '"' + p + '":'
            key_encoded = self.encode(key)[0].tolist()
            input_ids.extend(key_encoded)
            # Iteratively generate the parameter value, checking against
            # expected type
            r = ""  # clear tracked generation output
            if t["type"] == "string":
                # For strings, prepend the opening quote
                a = self.encode('"')[0].tolist()
                input_ids.extend(a)

            iteration_count = 0
            max_iterations = 30  # Prevent infinite loops
            # Copy current input_ids to track generation for this parameter
            ids = input_ids.copy()
            while iteration_count < max_iterations:
                iteration_count += 1
                if t["type"] == "number" or t["type"] == "integer":
                    logits = self.get_logits_from_input_ids(ids)
                    next_token_id = int(np.argmax(logits))
                    token_str = self.decode([next_token_id])
                    # Clean up strange spaces the tokenizer might introduce
                    clean_str = token_str.strip()
                    clean_str = clean_str.replace('"', "")
                    # 2. Check if the token is part of a valid number
                    # (Allowing '.' and '-' for decimals and negatives)
                    is_valid = all(c.isdigit() or c in '.-' for c in clean_str)
                    if is_valid:
                        # Simply add the token as-is, don't re-encode
                        ids.append(next_token_id)
                        r += clean_str
                    else:
                        # if parameter type if float, convert to float
                        if t["type"] != "integer":
                            new_n = float(r) if r else 0.0
                        else:
                            new_n = int(r)
                        next_token_id = self.encode(str(new_n))[0].tolist()
                        input_ids.extend(next_token_id)
                        break
                if t["type"] == "string":
                    logits = self.get_logits_from_input_ids(input_ids)
                    next_token_id = int(np.argmax(logits))
                    token_str = self.decode([next_token_id])
                    
                    # For strings, look for unescaped closing quote as the only stop marker
                    # This allows commas, braces, and other chars inside string values
                    print(token_str)
                    if '"' in token_str:

                        # Check if quote is escaped by counting preceding backslashes
                        idx = token_str.find('"')
                        num_backslashes = 0
                        j = idx - 1
                        while j >= 0 and token_str[j] == '\\':
                            num_backslashes += 1
                            j -= 1
                        # If even number of backslashes (including 0), quote is NOT escaped
                        if num_backslashes % 2 == 0:
                            # Found closing quote - split at it
                            token_str = token_str[:idx]
                            if token_str:
                                safe_token_str = escape_json_string_fragment(token_str)
                                encoded_token = self.encode(safe_token_str)[0].tolist()
                                r += safe_token_str
                                input_ids.extend(encoded_token)
                            # Add closing quote
                            input_ids.extend(self.encode('"')[0].tolist())
                            break

                    # No closing quote found, add entire token as part of string content
                    if iteration_count == 1 and token_str.startswith(" "):
                        token_str = token_str.lstrip()

                    if token_str:
                        safe_token_str = escape_json_string_fragment(token_str)
                        encoded_token = self.encode(safe_token_str)[0].tolist()
                        r += safe_token_str
                        input_ids.extend(encoded_token)

                if t["type"] == "boolean":
                    # For booleans, we expect the model to generate 'true' or 'false'
                    logits = self.get_logits_from_input_ids(input_ids)
                    next_token_id = int(np.argmax(logits))
                    token_str = self.decode([next_token_id])
                    clean_bool = token_str.lower().strip()
                    if clean_bool in ['true', 'false']:
                        # Encode and add the boolean value
                        encoded_bool = self.encode(clean_bool)[0].tolist()
                        input_ids.extend(encoded_bool)
                        break
                    else:
                        # If invalid boolean, try next iteration or timeout
                        if iteration_count >= max_iterations:
                            # Fallback to 'false' if no valid boolean found
                            fallback = self.encode('false')[0].tolist()
                            input_ids.extend(fallback)
            if p != list(params.keys())[-1]:
                # If it is not the last parameter, add a comma separator
                a = self.encode(', ')[0].tolist()
                input_ids.extend(a)

        # Close 'parameters' and the main JSON object blocks
        input_ids.append(self.id_json_close)
        input_ids.append(self.id_json_close)

        # Decode ONLY what was generated (ignoring the input prompt prefix)
        generated_text = self.decode(input_ids[prompt_len:])
        return generated_text.strip()

    def call_tool(self) -> None:
        start = time.perf_counter()
        results = []
        try:
            tests = load_json_file(self.tests_path)
        except FileNotFoundError:
            print(f"File not found: {self.tests_path}")
            return
        tools_ids = self.encode(self.tools_description)[0].tolist()
        for call in tests:
            prompt_text = call["prompt"]
            print(f"User request: {prompt_text}")
            prompt_str = "\n\nUser request: " + prompt_text
            prompt_ids = self.encode(prompt_str)[0].tolist()
            input_ids = tools_ids + prompt_ids
            raw = self.generate(input_ids, prompt_text)
            try:
                results.append(json.loads(raw))
                print(f"Generated JSON: {raw}\n")
            except json.JSONDecodeError as e:
                print(raw)
                results.append({
                    "prompt": prompt_text,
                    "error": f"invalid JSON: {e}",
                    "raw": raw
                })
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        except FileNotFoundError:
            print(f"File not found: {self.output_path}")
            return

        elapsed = time.perf_counter() - start
        print(f"Tempo de execução: {elapsed:.4f}s")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Call Me Maybe tool-calling assistant to generate function" +
            " calls based on user requests and function definitions."
        )
    )
    parser.add_argument(
        "--definitions-path",
        default=str(DEFAULT_FUNCTIONS_DEFS_PATH),
    )
    parser.add_argument(
        "--tests-path",
        default=str(DEFAULT_TESTS_PATH),
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
    )
    return parser


# export HF_HOME=/sgoinfre/$(whoami)/hf_cache
if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()
    call_me_maybe = CallMeMaybe(
        definitions_path=args.definitions_path,
        tests_path=args.tests_path,
        output_path=args.output_path,
    )
    call_me_maybe.call_tool()
"""
Commands from CallMeMaybe/moulinette:

teste1-public-generate:
uv run --active python ../data/call_me_maybe.py \
    --definitions-path ../data/input/public_functions_definition.json \
    --tests-path ../data/input/public_function_calling_tests.json \
    --output-path ../data/output/function_calling_results_public.json

teste1-public-grade:
uv run --active python -m moulinette grade_student_answers \
    ../data/output/function_calling_results_public.json --set public

teste2-private-generate:
uv run --active python ../data/call_me_maybe.py \
    --definitions-path ../data/input/private_functions_definition.json \
    --tests-path ../data/input/private_function_calling_tests.json \
    --output-path ../data/output/function_calling_results_private.json

teste2-private-grade:
uv run --active python -m moulinette grade_student_answers \
    ../data/output/function_calling_results_private.json --set private
"""
