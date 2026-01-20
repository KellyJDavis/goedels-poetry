"""Tests for proof composition with deep nested decomposition."""

from contextlib import suppress

import pytest

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
from goedels_poetry.state import GoedelsPoetryState

# Mark tests that require Kimina server as integration tests
# These tests use reconstruct_complete_proof(server_url=kimina_server_url) which now requires Kimina server for validation
pytestmark = pytest.mark.usefixtures("skip_if_no_lean")


def with_default_preamble(body: str) -> str:
    return combine_preamble_and_body(DEFAULT_IMPORTS, body)


def _create_ast_for_sketch(
    sketch: str, preamble: str = DEFAULT_IMPORTS, server_url: str = "http://localhost:8000", server_timeout: int = 60
):
    """
    Helper function to create an AST for a proof sketch in tests.

    This is needed because the new reconstruction implementation requires ASTs
    for all DecomposedFormalTheoremState nodes.
    """
    from kimina_client import KiminaClient

    from goedels_poetry.agents.util.common import combine_preamble_and_body, remove_default_imports_from_ast
    from goedels_poetry.agents.util.kimina_server import parse_kimina_ast_code_response
    from goedels_poetry.parsers.ast import AST

    # Normalize sketch and combine with preamble
    normalized_sketch = sketch.strip()
    normalized_sketch = normalized_sketch if normalized_sketch.endswith("\n") else normalized_sketch + "\n"
    normalized_preamble = preamble.strip()
    full_text = combine_preamble_and_body(normalized_preamble, normalized_sketch)

    # Calculate body_start
    if not normalized_preamble:
        body_start = 0
    elif not normalized_sketch:
        body_start = len(full_text)
    else:
        body_start = len(normalized_preamble) + 2  # +2 for "\n\n"

    # Parse with Kimina
    client = KiminaClient(api_url=server_url, n_retries=3, http_timeout=server_timeout)
    ast_response = client.ast_code(full_text, timeout=server_timeout)
    parsed = parse_kimina_ast_code_response(ast_response)

    if parsed.get("error") is not None:
        raise ValueError(f"Failed to parse sketch for AST: {parsed['error']}")  # noqa: TRY003

    ast_without_imports = remove_default_imports_from_ast(parsed["ast"], preamble=preamble)
    return AST(ast_without_imports, sorries=parsed.get("sorries"), source_text=full_text, body_start=body_start)


def _annotate_hole_offsets(  # noqa: C901
    node: dict,
    sketch: str,
    *,
    hole_name: str,
    anchor: str | None = None,
    occurrence: int = 0,
) -> None:
    """
    Attach AST-style hole metadata (hole_name/hole_start/hole_end) to a proof-tree node for tests.

    These tests construct proof trees manually (without a Kimina AST), so we compute the `sorry`
    span by simple string offsets. Production code computes these from the AST.
    """

    def _find_sorry_token(text: str, start: int) -> int:
        """
        Find the next standalone `sorry` token at/after `start`.

        This intentionally skips occurrences inside comments/strings (e.g. `-- ... 'sorry' ...`)
        by requiring whitespace boundaries around the word.
        """
        i = start
        while True:
            i = text.find("sorry", i)
            if i == -1:
                raise ValueError("No standalone `sorry` token found")  # noqa: TRY003
            before = text[i - 1] if i > 0 else " "
            after = text[i + len("sorry")] if i + len("sorry") < len(text) else " "
            if before.isspace() and after.isspace():
                return i
            i += len("sorry")

    def _nth_sorry_token(text: str, start: int, n: int) -> int:
        i = start
        for _ in range(n + 1):
            i = _find_sorry_token(text, i)
            i += len("sorry")
        return i - len("sorry")

    if anchor is None and hole_name == "<main body>":
        # In sketches, the main-body `sorry` is typically the last `sorry` token in the body.
        # Use the last standalone token, not a mention inside a comment.
        positions: list[int] = []
        cursor = 0
        while True:
            try:
                pos = _find_sorry_token(sketch, cursor)
            except ValueError:
                break
            positions.append(pos)
            cursor = pos + len("sorry")
        if not positions:
            raise ValueError("No standalone `sorry` token found for <main body>")  # noqa: TRY003
        start = positions[-1]
    else:
        base = 0 if anchor is None else sketch.index(anchor)
        start = _nth_sorry_token(sketch, base, occurrence)
    end = start + len("sorry")

    node["hole_name"] = hole_name
    node["hole_start"] = start
    node["hole_end"] = end


def test_reconstruct_complete_proof_deep_nested_decomposition_4_levels(kimina_server_url: str) -> None:
    """Test reconstruct_complete_proof with 4 levels of nested DecomposedFormalTheoremState."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    # Use True instead of P, A, B, C, D to avoid type variable issues
    theorem_sig = f"theorem test_deep_4_{uuid.uuid4().hex} : True"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Level 0: Root
        root_sketch = f"""{theorem_sig} := by
  have a : True := by sorry
  trivial"""
        root_ast = _create_ast_for_sketch(root_sketch, DEFAULT_IMPORTS, kimina_server_url)
        root = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=root_sketch,
            syntactic=True,
            errors=None,
            ast=root_ast,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Level 1
        level1_sketch = """lemma a : True := by
  have b : True := by sorry
  exact b"""
        level1_ast = _create_ast_for_sketch(level1_sketch, DEFAULT_IMPORTS, kimina_server_url)
        level1 = DecomposedFormalTheoremState(
            parent=cast(TreeNode, root),
            children=[],
            depth=1,
            formal_theorem="lemma a : True",
            preamble=DEFAULT_IMPORTS,
            proof_sketch=level1_sketch,
            syntactic=True,
            errors=None,
            ast=level1_ast,
            self_correction_attempts=1,
            decomposition_history=[],
        )
        _annotate_hole_offsets(level1, root_sketch, hole_name="a", anchor="have a")

        # Level 2
        level2_sketch = """lemma b : True := by
  have c : True := by sorry
  exact c"""
        level2_ast = _create_ast_for_sketch(level2_sketch, DEFAULT_IMPORTS, kimina_server_url)
        level2 = DecomposedFormalTheoremState(
            parent=cast(TreeNode, level1),
            children=[],
            depth=2,
            formal_theorem="lemma b : True",
            preamble=DEFAULT_IMPORTS,
            proof_sketch=level2_sketch,
            syntactic=True,
            errors=None,
            ast=level2_ast,
            self_correction_attempts=1,
            decomposition_history=[],
        )
        _annotate_hole_offsets(level2, level1_sketch, hole_name="b", anchor="have b")

        # Level 3
        level3_sketch = """lemma c : True := by
  have d : True := by sorry
  exact d"""
        level3_ast = _create_ast_for_sketch(level3_sketch, DEFAULT_IMPORTS, kimina_server_url)
        level3 = DecomposedFormalTheoremState(
            parent=cast(TreeNode, level2),
            children=[],
            depth=3,
            formal_theorem="lemma c : True",
            preamble=DEFAULT_IMPORTS,
            proof_sketch=level3_sketch,
            syntactic=True,
            errors=None,
            ast=level3_ast,
            self_correction_attempts=1,
            decomposition_history=[],
        )
        _annotate_hole_offsets(level3, level2_sketch, hole_name="c", anchor="have c")

        # Level 4: Leaf
        leaf = FormalTheoremProofState(
            parent=cast(TreeNode, level3),
            depth=4,
            formal_theorem="lemma d : True",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="trivial",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(leaf, str(level3["proof_sketch"]), hole_name="d", anchor="have d")

        # Build tree
        level3["children"].append(cast(TreeNode, leaf))
        level2["children"].append(cast(TreeNode, level3))
        level1["children"].append(cast(TreeNode, level2))
        root["children"].append(cast(TreeNode, level1))
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof(server_url=kimina_server_url)

        assert result.startswith(DEFAULT_IMPORTS)
        assert "have a : True" in result
        assert "have b : True" in result
        assert "have c : True" in result
        assert "have d : True" in result
        assert "trivial" in result
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)
