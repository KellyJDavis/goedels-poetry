"""Robust structural extraction of proof bodies and preambles from Lean 4 ASTs."""

from goedels_poetry.parsers.ast import AST
from goedels_poetry.parsers.util.foundation.ast_to_code import _ast_to_code
from goedels_poetry.parsers.util.foundation.kind_utils import __is_theorem_or_lemma_kind, __normalize_kind


def _is_comment_kind(kind: str) -> bool:
    """True if the node kind indicates a comment (docstring, block, or line comment)."""
    if not isinstance(kind, str):
        return False
    k = kind.lower()
    return "doccomment" in k or "comment" in k


def _reconstruct_signature_from_decl_node(decl_node: dict, *, skip_comments: bool = True) -> str:
    """
    Reconstruct the declaration signature (up to but not including ':=') from a decl AST node.

    Iterates decl_node.args, optionally skips comment-like nodes, stops at := / declValSimple /
    declValEqns, and joins reconstructed text. Recurses into 'group' args (e.g. lemma) so we
    stop at declVal/:= inside them. Used for both signature extraction and proof-body matching.
    """

    def collect(args: list, parts: list[str]) -> bool:
        """Append signature parts from args; return True if we stopped at :=/declVal."""
        for arg in args:
            if not isinstance(arg, dict):
                parts.append(_ast_to_code(arg))
                continue
            kind = arg.get("kind", "")
            if skip_comments and _is_comment_kind(kind):
                continue
            if kind in ("Lean.Parser.Command.declValSimple", "Lean.Parser.Command.declValEqns"):
                return True
            if "declval" in kind.lower():
                return True
            if arg.get("val") == ":=":
                return True
            if kind == "group":
                if collect(arg.get("args", []), parts):
                    return True
                continue
            parts.append(_ast_to_code(arg))
        return False

    out: list[str] = []
    collect(decl_node.get("args", []), out)
    return "".join(out).strip()


def extract_signature_from_ast(ast: AST) -> str | None:
    """
    Return the signature only (e.g. `theorem n : True`) for the last theorem/lemma in the AST.

    No comments, no `:= by sorry` or variants. Returns None if no such declaration exists.
    Reuses the same traversal and last-occurrence heuristic as extract_proof_body_from_ast.
    """
    ast_root = ast.get_ast()
    if not isinstance(ast_root, dict):
        return None

    commands = ast_root.get("commands", [])
    if not commands:
        commands = ast_root.get("args", [])

    last_decl_node: dict | None = None
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        target_node = cmd
        kind = cmd.get("kind", "")
        if kind == "Lean.Parser.Command.declaration":
            for arg in cmd.get("args", []):
                if isinstance(arg, dict) and __is_theorem_or_lemma_kind(arg.get("kind")):
                    target_node = arg
                    break
        if __is_theorem_or_lemma_kind(target_node.get("kind")):
            last_decl_node = target_node

    if last_decl_node is None:
        return None
    return _reconstruct_signature_from_decl_node(last_decl_node, skip_comments=True)


def extract_proof_body_from_ast(ast: AST, target_signature: str) -> str | None:  # noqa: C901
    """
    Finds the proof body (tactics) of a declaration matching the target signature.

    This function navigates the AST to find the declaration that matches the
    provided signature string, identifies its proof body structurally (looking for ':=' then 'by'),
    and extracts the tactics.

    If multiple declarations match, the last one is selected (last-occurrence heuristic).

    Parameters
    ----------
    ast : AST
        The Kimina AST of the complete Lean file.
    target_signature : str
        The signature string to match (e.g. "theorem t : True").

    Returns
    -------
    str | None
        The extracted tactics string (without the 'by' keyword), or None if not found.
    """
    ast_root = ast.get_ast()
    if not isinstance(ast_root, dict):
        return None

    # We look for all matching declarations and take the last one
    matching_proof_node = None

    # Traverse top-level commands
    commands = ast_root.get("commands", [])
    if not commands:
        commands = ast_root.get("args", [])

    for cmd in commands:
        if not isinstance(cmd, dict):
            continue

        # Handle Lean.Parser.Command.declaration wrapper
        target_node = cmd
        kind = cmd.get("kind", "")
        if kind == "Lean.Parser.Command.declaration":
            # Inner declaration is usually the second argument after modifiers
            args = cmd.get("args", [])
            for arg in args:
                if isinstance(arg, dict) and __is_theorem_or_lemma_kind(arg.get("kind")):
                    target_node = arg
                    break

        # Check for signature match
        # 1. Try Kimina's 'type' field if it exists
        decl_type = target_node.get("type") or cmd.get("type")
        if decl_type and decl_type.strip() == target_signature.strip():
            proof_node = _find_proof_body_node_structurally(target_node)
            if proof_node:
                matching_proof_node = proof_node
                continue

        # 2. Fallback: Reconstruct signature text from AST and compare (skip_comments=True
        #    so both sides are normalised; see plan 1.3 / 7.6)
        reconstructed_sig = _reconstruct_signature_from_decl_node(target_node, skip_comments=True)
        if " ".join(reconstructed_sig.split()) == " ".join(target_signature.split()):
            proof_node = _find_proof_body_node_structurally(target_node)
            if proof_node:
                matching_proof_node = proof_node

    if matching_proof_node:
        return _extract_tactics_from_proof_node(matching_proof_node)

    return None


def _find_proof_body_node_structurally(decl_node: dict) -> dict | None:  # noqa: C901
    """
    Navigates a declaration's args to find the proof body node after ':=' and 'by'.
    """
    args = decl_node.get("args", [])

    # 1. The proof is usually inside declValSimple
    for arg in args:
        if not isinstance(arg, dict):
            continue

        if arg.get("kind") == "Lean.Parser.Command.declValSimple":
            # Inside declValSimple, we have ":=", then the term/proof
            inner_args = arg.get("args", [])
            seen_assign = False
            for iarg in inner_args:
                if not isinstance(iarg, dict):
                    continue
                if iarg.get("val") == ":=":
                    seen_assign = True
                    continue
                if seen_assign:
                    # This is the proof node
                    proof_node = _check_if_proof_node(iarg)
                    if proof_node:
                        return proof_node

    # 2. Fallback: Proof might be a direct child of the declaration's args (common in dummy ASTs)
    seen_assign = False
    for arg in args:
        if not isinstance(arg, dict):
            continue
        if arg.get("val") == ":=":
            seen_assign = True
            continue
        if seen_assign:
            proof_node = _check_if_proof_node(arg)
            if proof_node:
                return proof_node

    return None


def _check_if_proof_node(node: dict) -> dict | None:
    """Helper to check if a node is a byTactic or tacticSeq."""
    kind = node.get("kind", "")
    if kind == "Lean.Parser.Term.byTactic":
        # Ensure it actually has the 'by' token
        by_args = node.get("args", [])
        if any(isinstance(a, dict) and a.get("val") == "by" for a in by_args):
            return node
    elif kind == "Lean.Parser.Tactic.tacticSeq":
        return node
    return None


def _extract_tactics_from_proof_node(proof_node: dict) -> str:
    """
    Extracts the tactics string from a byTactic or tacticSeq node.
    """
    kind = proof_node.get("kind", "")
    if kind == "Lean.Parser.Term.byTactic":
        # byTactic -> args[0] = "by", args[1] = tactics (tacticSeq)
        by_args = proof_node.get("args", [])
        if len(by_args) >= 2:
            # trailing of "by" contains a \n and the indent of the next line
            next_line_indent = ""
            if by_args[0].get("val", "") == "by":
                info = by_args[0].get("info", {})
                trailing = info.get("trailing", "")
                next_line_indent = trailing.lstrip("\n")
            return next_line_indent + _ast_to_code(by_args[1]).strip()
    elif kind == "Lean.Parser.Tactic.tacticSeq":
        return _ast_to_code(proof_node).strip()

    return _ast_to_code(proof_node).strip()


def extract_preamble_from_ast(ast: AST) -> str:  # noqa: C901
    """
    Extracts and orders imports and open statements from the AST.
    """
    ast_root = ast.get_ast()
    if not isinstance(ast_root, dict):
        return ""

    imports = []
    opens = []

    # Collect nodes from both header and commands
    all_nodes = []
    header = ast_root.get("header")
    if isinstance(header, dict):
        # Flatten header["args"] which is a list of lists of nodes
        header_args = header.get("args", [])
        for arg in header_args:
            if isinstance(arg, list):
                all_nodes.extend(arg)
            elif isinstance(arg, dict):
                all_nodes.append(arg)

    commands = ast_root.get("commands", [])
    if not commands:
        commands = ast_root.get("args", [])
    all_nodes.extend(commands)

    for cmd in all_nodes:
        if not isinstance(cmd, dict):
            continue

        kind = cmd.get("kind", "")
        # Normalize kind to handle qualified/unqualified names
        norm_kind = __normalize_kind(kind)

        if norm_kind == "Lean.Parser.Module.import":
            imports.append(_ast_to_code(cmd).strip())
        elif norm_kind == "Lean.Parser.Command.open":
            opens.append(_ast_to_code(cmd).strip())

    # Combine with imports first
    preamble_parts = []
    unique_imports = []
    if imports:
        # Deduplicate imports
        seen_imports = set()
        for imp in imports:
            if imp not in seen_imports:
                unique_imports.append(imp)
                seen_imports.add(imp)
        preamble_parts.extend(unique_imports)

    if opens:
        # Deduplicate opens
        seen_opens = set()
        unique_opens = []
        for op in opens:
            if op not in seen_opens:
                unique_opens.append(op)
                seen_opens.add(op)

        if unique_imports:
            preamble_parts.append("")
        preamble_parts.extend(unique_opens)

    return "\n".join(preamble_parts).strip()
