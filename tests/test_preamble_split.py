from goedels_poetry.agents.util.common import split_preamble_and_body


def test_doc_comment_after_preamble_is_in_body() -> None:
    code = """
import Mathlib
open Nat

/-- Test theorem doc comment. -/
theorem sample : True := by
  trivial
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert "import Mathlib" in preamble
    assert "/-- Test theorem doc comment. -/" in body
    assert "/-- Test theorem doc comment. -/" not in preamble


def test_module_doc_comment_before_import_stays_in_preamble() -> None:
    code = """
/-!
This module level doc comment documents the file.
-/
import Mathlib

theorem module_example : True := by
  trivial
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert preamble.startswith("/-!")
    assert "import Mathlib" in preamble
    assert "/-!" not in body


def test_doc_comment_without_header_commands_goes_to_body() -> None:
    code = """
/-- Standalone theorem description. -/
theorem isolated : True := by
  trivial
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert preamble == ""
    assert body.startswith("/-- Standalone theorem description. -/")


def test_doc_comment_between_imports_stays_in_preamble() -> None:
    code = """
import Mathlib

/-- Module overview. -/
open Nat

theorem mid_comment : True := by
  trivial
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert "/-- Module overview. -/" in preamble
    assert "open Nat" in preamble
    assert "/-- Module overview. -/" not in body


def test_attribute_before_theorem_is_kept_in_body() -> None:
    code = """
import Mathlib

@[simp] theorem attr_example : True := by
  trivial
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert "import Mathlib" in preamble
    assert body.startswith("@[simp]")


def test_header_only_results_in_empty_body() -> None:
    code = """
import Mathlib
open Nat
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert body == ""
    assert "open Nat" in preamble


def test_private_modifier_kept_with_declaration() -> None:
    code = """
import Mathlib

private theorem hidden : True := by
  trivial
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert "import Mathlib" in preamble
    assert body.startswith("private theorem hidden")
    assert "private theorem hidden : True" in body


def test_syntax_command_before_theorem_stays_in_preamble() -> None:
    code = """
import Mathlib

syntax (name := strange) "strange" term:41 : term

theorem uses_syntax : True := by
  trivial
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert 'syntax (name := strange) "strange"' in preamble
    assert body.startswith("theorem uses_syntax")


def test_attribute_only_response_detected_as_body() -> None:
    code = """
@[simp] theorem attr_only : True := by
  trivial
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert preamble == ""
    assert body.startswith("@[simp] theorem attr_only")


def test_multiple_modifiers_stay_with_declaration() -> None:
    code = """
import Mathlib

private unsafe theorem combo : True := by
  trivial
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert "import Mathlib" in preamble
    assert body.startswith("private unsafe theorem combo")


def test_doc_comment_followed_by_attribute_is_body() -> None:
    code = """
import Mathlib

/-- Important lemma. -/
@[simp] theorem attributed : True := by
  trivial
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert "/-- Important lemma. -/" in body
    assert body.startswith("/-- Important lemma. -/\n@[simp]")


def test_comment_directly_before_attribute_is_body() -> None:
    code = """
import Mathlib

-- simplify this theorem
@[simp] theorem comment_attr : True := by
  trivial
"""
    preamble, body = split_preamble_and_body(code.strip())
    assert "-- simplify this theorem" in body
    assert body.startswith("-- simplify this theorem")
