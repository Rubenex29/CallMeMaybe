import json
import traceback
import time

from src.call_me_maybe import CallMeMaybe


class TestSuite:
    def __init__(self):
        print("Loading model for tests...")
        self.model = CallMeMaybe()
        start_time = time.perf_counter()

        elapsed = time.perf_counter() - start_time
        print("-" * 40)
        print(
            f"Summary: {self.passed} passed | {self.failed} failed "
            f"(Time: {elapsed:.2f}s)"
        )


if __name__ == "__main__":
    suite = TestSuite()
    suite.run_all()
