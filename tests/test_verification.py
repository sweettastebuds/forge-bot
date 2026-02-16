"""Tests for the verification module."""

from __future__ import annotations

from forge_bot.handlers.verification import (
    ProgressTracker,
    VerifyResult,
    check_hallucination,
    check_relevance,
    verify_response,
)
from forge_bot.tools.base import ToolResult


# -- check_relevance --


class TestCheckRelevance:
    def test_todo_always_relevant(self) -> None:
        assert check_relevance("todo", {"action": "set"}, "anything") is True

    def test_search_api_always_relevant(self) -> None:
        assert check_relevance("search_api", {"keyword": "x"}, "anything") is True

    def test_exec_common_commands_relevant(self) -> None:
        for cmd in ["ls -la", "grep -r pattern .", "git log", "cat file.py"]:
            assert check_relevance("exec", {"command": cmd}, "anything") is True

    def test_exec_python_relevant(self) -> None:
        assert check_relevance(
            "exec", {"command": "python -m pytest"}, "run the tests"
        ) is True

    def test_api_call_defaults_relevant(self) -> None:
        # Default: permissive
        assert check_relevance(
            "api_call", {"endpoint": "get_repo"}, "what is this repo?"
        ) is True

    def test_unknown_tool_defaults_relevant(self) -> None:
        assert check_relevance("new_tool", {}, "question") is True

    def test_empty_command_relevant(self) -> None:
        # Empty command still passes (permissive default)
        assert check_relevance("exec", {"command": ""}, "question") is True


# -- check_hallucination --


class TestCheckHallucination:
    def test_no_exec_results_no_issues(self) -> None:
        results = [
            ("search_api", ToolResult("search_api", True, "Found 3 endpoints")),
        ]
        assert check_hallucination("tests pass", results) == []

    def test_pass_claim_with_nonzero_exit(self) -> None:
        results = [
            (
                "exec",
                ToolResult(
                    "exec",
                    False,
                    "Exit code: 1\nstdout:\n2 failed, 3 passed\nstderr:\n",
                ),
            ),
        ]
        mismatches = check_hallucination("All tests pass successfully.", results)
        assert len(mismatches) == 1
        assert "exit code 1" in mismatches[0]

    def test_pass_claim_with_zero_exit(self) -> None:
        results = [
            (
                "exec",
                ToolResult("exec", True, "Exit code: 0\nstdout:\n5 passed\nstderr:\n"),
            ),
        ]
        assert check_hallucination("All tests pass.", results) == []

    def test_fail_claim_with_zero_exit(self) -> None:
        results = [
            (
                "exec",
                ToolResult("exec", True, "Exit code: 0\nstdout:\nOK\nstderr:\n"),
            ),
        ]
        mismatches = check_hallucination("The tests fail with errors.", results)
        assert len(mismatches) == 1
        assert "exit code 0" in mismatches[0]

    def test_fail_claim_with_nonzero_exit(self) -> None:
        results = [
            (
                "exec",
                ToolResult("exec", False, "Exit code: 2\nstdout:\nFAILED\nstderr:\n"),
            ),
        ]
        assert check_hallucination("Tests fail.", results) == []

    def test_no_exit_code_in_output(self) -> None:
        results = [
            ("exec", ToolResult("exec", True, "hello world")),
        ]
        # No exit code to compare, so no mismatches
        assert check_hallucination("tests pass", results) == []

    def test_no_errors_claim_with_nonzero(self) -> None:
        results = [
            ("exec", ToolResult("exec", False, "Exit code: 1\nstdout:\nerror\n")),
        ]
        mismatches = check_hallucination("No errors found in the code.", results)
        assert len(mismatches) == 1

    def test_multiple_exec_results(self) -> None:
        results = [
            ("exec", ToolResult("exec", True, "Exit code: 0\nstdout:\nOK\n")),
            ("exec", ToolResult("exec", False, "Exit code: 1\nstdout:\nFAIL\n")),
        ]
        mismatches = check_hallucination("All tests succeed.", results)
        assert len(mismatches) == 1  # second exec contradicts


# -- ProgressTracker --


class TestProgressTracker:
    def test_no_duplicates_initially(self) -> None:
        tracker = ProgressTracker()
        assert tracker.is_duplicate("exec", "exec:ls") is False

    def test_not_duplicate_on_first_record(self) -> None:
        tracker = ProgressTracker()
        tracker.record("exec", "exec:ls")
        # First occurrence — not a duplicate yet
        assert tracker.is_duplicate("exec", "exec:ls") is False

    def test_duplicate_after_two_records(self) -> None:
        tracker = ProgressTracker()
        tracker.record("exec", "exec:ls")
        tracker.record("exec", "exec:ls")
        assert tracker.is_duplicate("exec", "exec:ls") is True

    def test_different_args_not_duplicate(self) -> None:
        tracker = ProgressTracker()
        tracker.record("exec", "exec:ls")
        tracker.record("exec", "exec:pwd")
        assert tracker.is_duplicate("exec", "exec:ls") is False

    def test_not_stuck_initially(self) -> None:
        tracker = ProgressTracker()
        assert tracker.is_stuck() is False

    def test_not_stuck_with_new_calls(self) -> None:
        tracker = ProgressTracker()
        tracker.record_round(had_new_calls=True)
        tracker.record_round(had_new_calls=True)
        assert tracker.is_stuck() is False

    def test_stuck_after_two_stale_rounds(self) -> None:
        tracker = ProgressTracker()
        tracker.record_round(had_new_calls=False)
        tracker.record_round(had_new_calls=False)
        assert tracker.is_stuck() is True

    def test_reset_stale_on_new_call(self) -> None:
        tracker = ProgressTracker()
        tracker.record_round(had_new_calls=False)
        tracker.record_round(had_new_calls=True)  # resets
        tracker.record_round(had_new_calls=False)
        assert tracker.is_stuck() is False

    def test_stuck_after_reset_and_two_stale(self) -> None:
        tracker = ProgressTracker()
        tracker.record_round(had_new_calls=True)
        tracker.record_round(had_new_calls=False)
        tracker.record_round(had_new_calls=False)
        assert tracker.is_stuck() is True


# -- verify_response --


class TestVerifyResponse:
    def test_clean_response(self) -> None:
        results = [
            ("exec", ToolResult("exec", True, "Exit code: 0\nstdout:\nOK\n")),
        ]
        vr = verify_response("Everything looks good. Tests pass.", results)
        assert vr.passed is True
        assert vr.failures == []

    def test_hallucination_detected(self) -> None:
        results = [
            ("exec", ToolResult("exec", False, "Exit code: 1\nstdout:\nFAILED\n")),
        ]
        vr = verify_response("All tests pass.", results)
        assert vr.passed is False
        assert len(vr.failures) >= 1

    def test_fabricated_paths_detected(self) -> None:
        # Response references many paths not in any tool output
        response = (
            "I found issues in `src/auth/login.py`, `src/auth/oauth.py`, "
            "`src/models/user.py`, `src/api/routes.py`, and `lib/utils.py`. "
            "All of these files need fixing."
        )
        results = [
            ("exec", ToolResult("exec", True, "Exit code: 0\nstdout:\nhello\n")),
        ]
        vr = verify_response(response, results)
        assert vr.passed is False
        assert any("file paths" in f for f in vr.failures)

    def test_paths_seen_in_tool_output_ok(self) -> None:
        response = (
            "The file `src/main.py` has a bug on line 5. "
            "Also check `src/utils.py` and `tests/test_main.py`."
        )
        tool_output = (
            "Exit code: 0\nstdout:\n"
            "src/main.py\nsrc/utils.py\ntests/test_main.py\n"
            "src/config.py\n"
        )
        results = [("exec", ToolResult("exec", True, tool_output))]
        vr = verify_response(response, results)
        assert vr.passed is True

    def test_common_paths_not_counted_as_fabricated(self) -> None:
        # README.md, setup.py, main.py, index.js are excluded from fabrication check
        response = (
            "Check README.md, setup.py, main.py, and index.js for details."
        )
        results = [
            ("exec", ToolResult("exec", True, "Exit code: 0\nstdout:\nOK\n")),
        ]
        vr = verify_response(response, results)
        assert vr.passed is True

    def test_few_fabricated_paths_still_pass(self) -> None:
        # 3 or fewer fabricated paths is OK (threshold is >3)
        response = (
            "See `src/a.py`, `src/b.py`, and `src/c.py` for the changes."
        )
        results = [
            ("exec", ToolResult("exec", True, "Exit code: 0\nstdout:\nOK\n")),
        ]
        vr = verify_response(response, results)
        assert vr.passed is True

    def test_empty_results(self) -> None:
        vr = verify_response("Here is my response.", [])
        assert vr.passed is True

    def test_verify_result_dataclass(self) -> None:
        vr = VerifyResult(passed=False, failures=["oops"])
        assert vr.passed is False
        assert vr.failures == ["oops"]
