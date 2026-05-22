import json
import traceback
import time
from call_me_maybe import CallMeMaybe


class TestSuite:
    def __init__(self):
        print("Loading model for tests...")
        self.model = CallMeMaybe()
        self.tools_ids = (
            self.model.encode(self.model.tools_description)[0].tolist()
        )
        self.passed = 0
        self.failed = 0

    def simulate_prompt(self, prompt_text):
        """Helper to generate JSON output from a prompt."""
        prompt_str = "\n\nUser request: " + prompt_text
        prompt_ids = self.model.encode(prompt_str)[0].tolist()
        input_ids = self.tools_ids + prompt_ids

        raw_output = self.model.generate(input_ids, prompt_text)
        return json.loads(raw_output)

    def assert_function_name(self, result, expected_name):
        assert result["name"] == expected_name, (
            f"Expected function '{expected_name}', got '{result.get('name')}'"
        )

    def assert_valid_tool_call(self, result):
        assert isinstance(result, dict), "Model output must be a JSON object"
        assert "name" in result, "Missing top-level key 'name'"
        assert "parameters" in result, "Missing top-level key 'parameters'"

        expected_param_keys = {
            "fn_add_numbers": {"a", "b"},
            "fn_greet": {"name"},
            "fn_reverse_string": {"s"},
            "fn_get_square_root": {"a"},
            "fn_substitute_string_with_regex": {
                "source_string",
                "regex",
                "replacement",
            },
        }

        tool_name = result["name"]
        assert tool_name in expected_param_keys, f"Unknown function name '{tool_name}'"

        params = result["parameters"]
        assert isinstance(params, dict), "'parameters' must be an object"
        missing_keys = expected_param_keys[tool_name] - set(params.keys())
        assert not missing_keys, f"Missing parameter keys: {sorted(missing_keys)}"

    # --- TESTS: fn_add_numbers ---

    def test_add_numbers_simple_integers(self):
        result = self.simulate_prompt("What is the sum of 265 and 345?")
        self.assert_function_name(result, "fn_add_numbers")

    def test_add_numbers_negative_and_positive(self):
        result = self.simulate_prompt("What is the sum of -5 and 10?")
        self.assert_function_name(result, "fn_add_numbers")
        params = result["parameters"]
        assert -5 in params.values(), "Failed to extract negative value -5"

    def test_add_numbers_decimals(self):
        result = self.simulate_prompt("Add 3.14 to 2")
        self.assert_function_name(result, "fn_add_numbers")
        params = result["parameters"]
        assert 3.14 in params.values(), "Failed to extract decimal value 3.14"

    def test_add_numbers_zero(self):
        result = self.simulate_prompt("Add 0 and 999")
        self.assert_function_name(result, "fn_add_numbers")

    def test_add_numbers_large_numbers(self):
        result = self.simulate_prompt("Please add 1000000 and 999999")
        self.assert_function_name(result, "fn_add_numbers")

    # --- TESTS: fn_greet ---

    def test_greet_simple_name(self):
        result = self.simulate_prompt("Greet john")
        self.assert_function_name(result, "fn_greet")
        assert "name" in result["parameters"], "Missing parameter 'name'"

    def test_greet_uppercase_name(self):
        result = self.simulate_prompt("Greet MARIA")
        self.assert_function_name(result, "fn_greet")

    def test_greet_name_with_number(self):
        result = self.simulate_prompt("Greet user42")
        self.assert_function_name(result, "fn_greet")

    def test_greet_name_with_hyphen(self):
        result = self.simulate_prompt("Please greet ana-maria")
        self.assert_function_name(result, "fn_greet")

    # --- TESTS: fn_reverse_string ---

    def test_reverse_string_special_characters(self):
        result = self.simulate_prompt("Reverse the string '#%@&*!'")
        self.assert_function_name(result, "fn_reverse_string")
        assert result["parameters"].get("s") == "#%@&*!", "Wrong 's' extraction"

    def test_reverse_string_word(self):
        result = self.simulate_prompt("Reverse the string 'world'")
        self.assert_function_name(result, "fn_reverse_string")

    def test_reverse_string_with_spaces(self):
        result = self.simulate_prompt("Reverse the string 'hello world'")
        self.assert_function_name(result, "fn_reverse_string")

    def test_reverse_string_with_numbers(self):
        result = self.simulate_prompt("Reverse the string 'abc123'")
        self.assert_function_name(result, "fn_reverse_string")

    def test_reverse_string_empty(self):
        result = self.simulate_prompt("Reverse the string ''")
        self.assert_function_name(result, "fn_reverse_string")

    # --- TESTS: fn_get_square_root ---

    def test_square_root_perfect_number(self):
        result = self.simulate_prompt("What is the square root of 16?")
        self.assert_function_name(result, "fn_get_square_root")
        assert "a" in result["parameters"], "Missing parameter 'a'"

    def test_square_root_large_perfect_number(self):
        result = self.simulate_prompt("Calculate the square root of 144")
        self.assert_function_name(result, "fn_get_square_root")

    def test_square_root_decimal(self):
        result = self.simulate_prompt("What is the square root of 2.25?")
        self.assert_function_name(result, "fn_get_square_root")

    def test_square_root_one(self):
        result = self.simulate_prompt("Square root of 1")
        self.assert_function_name(result, "fn_get_square_root")

    # --- TESTS: fn_substitute_string_with_regex ---

    def test_substitute_regex_numbers(self):
        result = self.simulate_prompt(
            "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
        )
        self.assert_function_name(result, "fn_substitute_string_with_regex")
        params = result["parameters"]
        for key in ("source_string", "regex", "replacement"):
            assert key in params, f"Missing parameter '{key}'"

    def test_substitute_regex_vowels(self):
        result = self.simulate_prompt(
            "Replace all vowels in 'Programming is fun' with asterisks"
        )
        self.assert_function_name(result, "fn_substitute_string_with_regex")

    def test_substitute_regex_word(self):
        result = self.simulate_prompt(
            "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"
        )
        self.assert_function_name(result, "fn_substitute_string_with_regex")

    def test_substitute_regex_whitespace(self):
        result = self.simulate_prompt(
            "Replace every whitespace in 'a b   c' with '-'"
        )
        self.assert_function_name(result, "fn_substitute_string_with_regex")

    def test_substitute_regex_email(self):
        result = self.simulate_prompt(
            "Replace all domains in 'alice@example.com and bob@test.org' with hidden-domain"
        )
        self.assert_function_name(result, "fn_substitute_string_with_regex")

    # --- MIXED / ROBUSTNESS TESTS ---

    def test_add_numbers_short_format(self):
        result = self.simulate_prompt("2 + 8")
        self.assert_function_name(result, "fn_add_numbers")

    def test_greet_formal(self):
        result = self.simulate_prompt("Could you greet Alice please?")
        self.assert_function_name(result, "fn_greet")

    def test_reverse_string_with_punctuation(self):
        result = self.simulate_prompt("Reverse the string 'Wait... what?!'")
        self.assert_function_name(result, "fn_reverse_string")

    def test_square_root_natural_phrase(self):
        result = self.simulate_prompt("Can you compute the square root of 49 for me?")
        self.assert_function_name(result, "fn_get_square_root")

    def test_substitute_regex_natural_phrase(self):
        result = self.simulate_prompt(
            "Please replace all digits in 'Order 123, item 45' with X"
        )
        self.assert_function_name(result, "fn_substitute_string_with_regex")

    # --- STRESS TESTS: AMBIGUITY AND LONG STRINGS ---

    def test_stress_ambiguous_add_or_square_root(self):
        result = self.simulate_prompt("Can you process 49 and maybe add it to 1?")
        self.assert_valid_tool_call(result)
        assert result["name"] in {"fn_add_numbers", "fn_get_square_root"}, (
            "Ambiguous prompt should map to add_numbers or get_square_root"
        )

    def test_stress_ambiguous_greet_or_reverse(self):
        result = self.simulate_prompt("Greet and then reverse the string 'Alice'")
        self.assert_valid_tool_call(result)
        assert result["name"] in {"fn_greet", "fn_reverse_string"}, (
            "Ambiguous prompt should map to greet or reverse_string"
        )

    def test_stress_ambiguous_regex_or_reverse(self):
        result = self.simulate_prompt(
            "Reverse this and maybe replace vowels in 'Programming 123'"
        )
        self.assert_valid_tool_call(result)
        assert result["name"] in {
            "fn_reverse_string",
            "fn_substitute_string_with_regex",
        }, "Ambiguous prompt should map to reverse_string or substitute_string_with_regex"

    def test_stress_very_long_prompt_reverse(self):
        long_s = "A" * 1200 + "-END"
        result = self.simulate_prompt(f"Reverse the string '{long_s}'")
        self.assert_valid_tool_call(result)
        self.assert_function_name(result, "fn_reverse_string")

    def test_stress_very_long_prompt_greet(self):
        long_name = "User_" + ("x" * 900)
        result = self.simulate_prompt(f"Please greet {long_name}")
        self.assert_valid_tool_call(result)
        self.assert_function_name(result, "fn_greet")

    def test_stress_very_long_prompt_regex(self):
        long_text = "word " * 500
        result = self.simulate_prompt(
            f"Replace all spaces in '{long_text}' with '_'"
        )
        self.assert_valid_tool_call(result)
        self.assert_function_name(result, "fn_substitute_string_with_regex")

    def test_stress_escape_backslashes_and_quotes(self):
        result = self.simulate_prompt(
            "Replace every whitespace in 'C:\\Users\\name\\my file.txt' with '-'"
        )
        self.assert_valid_tool_call(result)
        self.assert_function_name(result, "fn_substitute_string_with_regex")

    def test_stress_long_with_symbols(self):
        noisy = ("@#$%^&*()[]{}<>?/|" * 70) + " done"
        result = self.simulate_prompt(f"Reverse the string '{noisy}'")
        self.assert_valid_tool_call(result)
        self.assert_function_name(result, "fn_reverse_string")

    def test_stress_very_long_ambiguous(self):
        long_block = "verylong " * 350
        result = self.simulate_prompt(
            f"Please do something with this text: '{long_block}'. Maybe greet, maybe reverse."
        )
        self.assert_valid_tool_call(result)
        assert result["name"] in {
            "fn_greet",
            "fn_reverse_string",
            "fn_substitute_string_with_regex",
        }, "Expected one plausible function for a very ambiguous long prompt"

    # --- TEST RUNNER ---

    def run_all(self):
        print("\nStarting Call Me Maybe Test Suite...\n" + "-" * 40)

        # Automatically discover all test methods.
        tests = []
        for name in dir(self):
            if name.startswith("test_"):
                method = getattr(self, name)
                if callable(method):
                    tests.append(method)
        tests.sort(key=lambda fn: fn.__name__)

        start_time = time.perf_counter()

        for test in tests:
            test_name = test.__name__
            try:
                test()
                print(f"[PASS] {test_name}")
                self.passed += 1
            except AssertionError as e:
                print(f"[FAIL] {test_name} - {e}")
                self.failed += 1
            except Exception as e:
                print(f"[ERROR] {test_name} - {type(e).__name__}: {e}")
                traceback.print_exc()
                self.failed += 1

        elapsed = time.perf_counter() - start_time
        print("-" * 40)
        print(
            f"Summary: {self.passed} passed | {self.failed} failed "
            f"(Time: {elapsed:.2f}s)"
        )


if __name__ == "__main__":
    suite = TestSuite()
    suite.run_all()
