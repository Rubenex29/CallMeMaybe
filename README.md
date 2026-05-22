*This project has been created as part of the 42 curriculum by rumontei.*

# Call Me Maybe

## Description
Call Me Maybe is a local tool-calling assistant built around a constrained decoding pipeline. Given a natural-language request and a set of function definitions, the model must emit a strict JSON object describing the function to call and the values for its parameters.

The goal of the project is not to generate free-form text, but to reliably map user intent to a valid function call that can be consumed by an automated grader. The implementation focuses on correctness, JSON validity, and deterministic behaviour under constrained output rules.

## Instructions
### Requirements
This project is written in Python and uses a local Hugging Face causal language model through the `llm_sdk` package. You need:

- Python 3.10 or newer
- `torch`
- `transformers`
- `huggingface-hub`
- `numpy`

If you are using the environment provided for the 42 evaluation, these dependencies are typically already available. No compilation step is required.

### Installation
Create and activate a virtual environment, then install the tooling used by the project or just run make install:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install uv
uv sync --directory llm_sdk --active --no-cache
python3 -m pip install mypy
python3 -m pip install flake8
python3 -m pip install accelerate
python3 -m pip install llm_sdk
```

The `accelerate` package is only needed when Transformers uses `device_map="auto"` on GPU-backed setups. On CPU or MPS-only runs it is optional, but keeping it installed avoids environment-specific failures. The separate `llm_sdk` install is also optional if the local `uv sync --directory llm_sdk --active --no-cache` step already provides the package in your environment.

If you are running on the 42 machines or another shared environment, it can also be useful to cache the model files on the expected storage path:

```bash
export HF_HOME=/sgoinfre/$(whoami)/hf_cache
```

### Execution
T. The safest way from the repository root is to pass explicit paths, because the defaults in `src/call_me_maybe.py` are relative to the current working directory.

```bash
uv run python3 src/call_me_maybe.py \
	--definitions-path data/input/functions_definition.json \
	--tests-path data/input/function_calling_tests.json \
	--output-path data/output/function_calling_results.json
```

## Algorithm Explanation
The core idea is constrained decoding over the model's token stream.

1. The function catalogue is loaded once and converted into a tool description prompt.
2. Fixed fragments of JSON syntax are pre-encoded, so the generator can append stable tokens directly instead of re-encoding the same text repeatedly.
3. The model is asked to predict the next token, but the logits are filtered so that only tokens compatible with the current schema state remain valid.
4. The first constrained phase selects the function name. At each step, the implementation keeps only tokens that preserve a valid prefix of one of the known function names.
5. Once the function is fixed, the decoder switches to parameter generation. Each parameter type is handled separately:
	 - strings are generated until an unescaped closing quote is found
	 - numbers are accepted only if the token stream stays numeric
	 - booleans are constrained to `true` or `false`
6. The final output is reconstructed as JSON and validated with `json.loads` before it is written to disk.

This is where encoding and decoding interact tightly with constrained decoding: encoding is used to map the prompt, fixed syntax, and accepted fragments into token IDs, while decoding is used to inspect model outputs and decide whether a token is valid under the current constraint set.

## Design Decisions
The implementation intentionally prefers deterministic behaviour over sampling.

- Greedy decoding with `argmax` keeps the output stable and easier to grade.
- Cached function definitions avoid re-reading the catalogue during every generation step.
- Pre-encoded JSON fragments reduce repeated tokenizer work for the fixed structure of the response.
- The generator validates each parameter by type instead of trusting the model to stay syntactically correct.
- String escaping is handled explicitly so that quotes, backslashes, and control characters do not break the JSON output.
- The model is allowed to generate only what is necessary; everything else is hard-coded or constrained.

## Performance Analysis
The solution is optimized for reliability first, then speed.

- Accuracy is improved by schema-aware masking: the model cannot drift into invalid function names or malformed parameter shapes.
- Speed is helped by caching the function catalogue and pre-encoding stable tokens such as the JSON skeleton.
- The implementation is batch-aware at the SDK layer, but constrained decoding itself remains sequential because each token depends on the previous one.
- Reliability is strong on the supported prompt set because the decoder rejects invalid structural tokens and falls back safely when a value cannot be completed cleanly.
- The main trade-off is latency: token-by-token filtering is slower than unconstrained generation, especially for long string parameters and large prompts.

## Challenges Faced
The main difficulties were keeping the output valid JSON while still allowing the model to express flexible string content.

- Maintaining the JSON format was essential, because any extra text or missing brace would fail grading.
- Generating strings was tricky because a quote may be part of the content or the actual closing delimiter.
- Numeric generation needed special handling for negatives, decimals, and fallback cases.
- Ambiguous prompts had to be handled without letting the model drift outside the allowed function set.
- The model sometimes attempted to produce noisy tokens, so the decoder had to filter aggressively and recover safely.

These issues were addressed by prefix-based token masking, explicit escaping, type-specific parsing, and fallback logic for invalid values.

## Testing Strategy
Validation is done with a broad custom suite in `src/test.py` and with the course grading flow.

The tests cover:

- exact function selection for each supported tool
- numbers, integers, decimals, negatives, and zero
- names with uppercase letters, digits, and hyphens
- strings with spaces, punctuation, empty content, and special characters
- regex-style replacement prompts
- ambiguous prompts that could map to more than one tool
- long prompts and noisy symbol-heavy inputs
- escaping edge cases with quotes and backslashes

This gives coverage over both correctness and robustness, especially for the constrained decoding layer where malformed output is the main failure mode.

## Example Usage
Generate a public result file:

```bash
uv run python3 src/call_me_maybe.py \
	--definitions-path data/input/public_functions_definition.json \
	--tests-path data/input/public_function_calling_tests.json \
	--output-path data/output/function_calling_results_public.json
```

Example of the expected output shape:

```json
{
	"name": "fn_greet",
	"parameters": {
		"name": "Alice"
	}
}
```

Grade the generated file using the provided course grader in the official environment.

## Resources
Classic references and documentation related to this project:

- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
- Hugging Face tokenizers and generation concepts: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- JSON specification overview: https://www.json.org/json-en.html
- Python `json` module documentation: https://docs.python.org/3/library/json.html
- Hugging Face Hub documentation: https://huggingface.co/docs/huggingface_hub

### AI Usage
AI was used to help structure this README, refine the wording, and ensure the required sections were present. The implementation, constrained decoding logic, test harness, and grading workflow were inspected directly from the project codebase and described based on the actual code paths in this repository.
