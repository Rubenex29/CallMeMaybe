import json
import time
import numpy as np

try:
    from ..llm_sdk import Small_LLM_Model
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from llm_sdk import Small_LLM_Model


FUNCTIONS_DEFS_PATH = "input/functions_definition.json"
TESTS_PATH = "input/function_calling_tests.json"
OUTPUT_PATH = "output/function_calling_results.json"
try:
    with open(FUNCTIONS_DEFS_PATH, "r", encoding="utf-8") as f:
        functions_list = json.load(f)
except FileNotFoundError:
    print(f"File not found: {FUNCTIONS_DEFS_PATH}")
    sys.exit(1)
# Initialize tools info string
tools_info = ""

for fn in functions_list:
    tools_info += f"Tool Name: {fn['name']}\n"
    tools_info += f"Description: {fn['description']}\n"
    tools_info += f"Parameters: {fn['parameters']}\n"
    tools_info += "\n"
TOOLS_DESCRIPTION = f"""
You are a tool-calling assistant.

If the user request can be solved with a tool,
respond ONLY with valid JSON.

The JSON should have the following format:
Example:
Every detail:
{tools_info}

Do not add text before or after the JSON.
It must end with the closing curly brace of the JSON.
"""


class CallMeMaybe(Small_LLM_Model):
    def __init__(self):
        super().__init__()
        self.id_json_open = self.encode("{")[0].tolist()[0]
        self.id_json_close = self.encode("}")[0].tolist()[0]
        self.id_quote = self.encode('"')[0].tolist()[0]
        self.id_colon = self.encode(":")[0].tolist()[0]

        # Load function definitions once to avoid repeated file openings
        self.functions = functions_list

        # Pre-encode fixed tokens to simply extend the input_ids without
        # running inference
        self.name_token = self.encode(' "name": "')[0].tolist()
        self.params_open_token = self.encode(', "parameters": {')[0].tolist()

    def get_func_definitions(self):
        # Return the cached functions loaded during __init__
        return self.functions

    def generate(self, input_ids: list, user_request: str) -> str:
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
        i = 0
        funcs = self.get_func_definitions()
        func_name = {}
        for f in funcs:
            func_name[f["name"]] = f["description"], f["parameters"]
        # Handle the case where the model needs to pick an exact function name
        while r not in func_name.keys():
            logits = self.get_logits_from_input_ids(input_ids)
            constrained_logits = np.full_like(logits, -float("inf"))
            for func in func_name.keys():
                if func.startswith(r):
                    func_encoded = self.encode(func)[0].tolist()[i]
                    constrained_logits[func_encoded] = logits[func_encoded]
            logits = constrained_logits
            i += 1
            next_token_id = int(np.argmax(logits))
            r += self.decode([next_token_id])
            input_ids.append(next_token_id)

        i = 0
        a = self.encode('"')[0].tolist()
        input_ids.extend(a)
        input_ids.extend(param)

        def get_function(name, functions):
            return next((fn for fn in functions if fn["name"] == name), None)

        fn = get_function(r, self.functions)
        params = fn["parameters"] if fn else {}

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
            while True:
                # 1. Get the natural LLM prediction logit (without -inf
                # constraint)
                logits = self.get_logits_from_input_ids(input_ids)
                next_token_id = int(np.argmax(logits))
                token_str = self.decode([next_token_id])

                if t["type"] == "number":
                    # Clean up strange spaces the tokenizer might introduce
                    # print(token_str)
                    clean_str = token_str.strip()

                    # 2. Check if the token is part of a valid number
                    # (Allowing '.' and '-' for decimals and negatives)
                    is_valid = all(c.isdigit() or c in '.-' for c in clean_str)
                    if is_valid:
                        input_ids.append(next_token_id)
                        r += token_str
                    else:
                        break
                else:
                    # Logic for strings: stop generation at a closing character
                    if '"' in token_str:
                        # remove quotes AND braces to avoid inserting them
                        cleaned = token_str.replace('"', '').replace('{', '').replace('}', '')
                        # also drop newlines inside string
                        cleaned = cleaned.replace('\n', '').replace('\r', '')
                        if cleaned:
                            cleaned_ids = self.encode(cleaned)[0].tolist()
                            input_ids.extend(cleaned_ids)
                            r += cleaned
                        # if token contained closing braces, stop
                        if '}' in token_str:
                            break
                        continue

                    if ('"}' in token_str or "{" in token_str or "}" in token_str or "\n" in token_str):
                        token_str = token_str.replace('{', '').replace('}', '').replace('\n', '')
                        r += token_str
                        break

                    input_ids.append(next_token_id)
                    r += token_str
                    print(r)
            if t["type"] == "string":
                a = self.encode('"')[0].tolist()
                input_ids.extend(a)
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

    def call_tool(self):
        start = time.perf_counter()
        results = []
        try:
            with open(TESTS_PATH, "r", encoding="utf-8") as f:
                tests = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {TESTS_PATH}")
            return
        tools_ids = self.encode(TOOLS_DESCRIPTION)[0].tolist()
        for call in tests:
            prompt_text = call["prompt"]
            print(f"User request: {prompt_text}")
            prompt_str = "\n\nUser request: " + prompt_text
            prompt_ids = self.encode(prompt_str)[0].tolist()
            input_ids = tools_ids + prompt_ids
            raw = self.generate(input_ids, prompt_text)
            print(raw)
            try:
                results.append(json.loads(raw))
                print(f"Generated JSON: {raw}")
            except json.JSONDecodeError as e:
                results.append({
                    "prompt": prompt_text,
                    "error": f"invalid JSON: {e}",
                    "raw": raw
                })
        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        except FileNotFoundError:
            print(f"File not found: {OUTPUT_PATH}")
            return

        elapsed = time.perf_counter() - start
        print(f"Tempo de execução: {elapsed:.4f}s")


# export HF_HOME=/sgoinfre/$(whoami)/hf_cache
if __name__ == "__main__":
    call_me_maybe = CallMeMaybe()
    call_me_maybe.call_tool()
