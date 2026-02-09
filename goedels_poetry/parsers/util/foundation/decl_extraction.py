"""Robust structural extraction of proof bodies and preambles from Lean 4 ASTs."""

import logging

from goedels_poetry.parsers.ast import AST
from goedels_poetry.parsers.util.foundation.ast_to_code import _ast_to_code
from goedels_poetry.parsers.util.foundation.kind_utils import __is_theorem_or_lemma_kind, __normalize_kind

logger = logging.getLogger(__name__)


def _ws_normalize(s: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(str(s).strip().split())


def _canonicalize_lemma_theorem(sig: str) -> str:
    """
    Canonicalize the declaration kind token for matching.

    For signature-matching purposes, `lemma` and `theorem` are semantically interchangeable in Lean.
    This helper replaces the first token equal to `lemma` or `theorem` (not necessarily the first
    token overall, to allow for leading modifiers/attributes) with `theorem` and whitespace-normalizes.
    """
    tokens = str(sig).strip().split()
    out: list[str] = []
    replaced = False
    for tok in tokens:
        if not replaced and tok in {"lemma", "theorem"}:
            out.append("theorem")
            replaced = True
        else:
            out.append(tok)
    return " ".join(out)


def _remove_space_before_closing_delims(s: str) -> str:
    """
    Remove whitespace immediately before closing delimiters.

    This defends against pretty-printing / token-trailing artifacts that can yield signatures like
    `(hx : 1 < x )` (note the space before `)`) that are semantically identical to `(hx : 1 < x)`,
    but fail token-based whitespace normalization.
    """
    out = str(s)
    for delim in (")", "]", "}"):
        out = out.replace(f" {delim}", delim)
    return out


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

    # Traverse top-level commands
    commands = ast_root.get("commands", [])
    if not commands:
        commands = ast_root.get("args", [])

    def _find_last_matching_proof_node(*, allow_lemma_theorem_equivalence: bool) -> tuple[dict | None, str | None]:  # noqa: C901
        """
        Return the last proof-node whose declaration matches the target signature.

        Stage A (allow_lemma_theorem_equivalence=False) preserves the existing strict matching:
        - Kimina `type` field: strip-only equality
        - Reconstructed signature: whitespace-normalized equality

        Stage B (allow_lemma_theorem_equivalence=True) relaxes only lemma/theorem differences by
        canonicalizing the first occurrence of the tokens `lemma`/`theorem` (wherever they occur)
        before comparison.
        """
        matching_proof_node: dict | None = None
        matched_sig: str | None = None

        if allow_lemma_theorem_equivalence:
            target_key = _canonicalize_lemma_theorem(target_signature)
        else:
            target_key = _ws_normalize(target_signature)

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
            # 1) Try Kimina's 'type' field if it exists.
            decl_type = target_node.get("type") or cmd.get("type")
            if decl_type:
                if allow_lemma_theorem_equivalence:
                    matches = _canonicalize_lemma_theorem(decl_type) == target_key
                else:
                    matches = str(decl_type).strip() == str(target_signature).strip()

                if matches:
                    proof_node = _find_proof_body_node_structurally(target_node)
                    if proof_node:
                        matching_proof_node = proof_node
                        matched_sig = str(decl_type)
                        continue

            # 2) Fallback: Reconstruct signature text from AST and compare (skip_comments=True
            #    so both sides are normalised; see plan 1.3 / 7.6)
            reconstructed_sig = _reconstruct_signature_from_decl_node(target_node, skip_comments=True)
            if allow_lemma_theorem_equivalence:
                matches = _canonicalize_lemma_theorem(reconstructed_sig) == target_key
            else:
                matches = " ".join(reconstructed_sig.split()) == " ".join(str(target_signature).split())

            if matches:
                proof_node = _find_proof_body_node_structurally(target_node)
                if proof_node:
                    matching_proof_node = proof_node
                    matched_sig = reconstructed_sig

        return matching_proof_node, matched_sig

    # Stage A: strict match.
    proof_node, _matched_sig = _find_last_matching_proof_node(allow_lemma_theorem_equivalence=False)
    if proof_node:
        return _extract_tactics_from_proof_node(proof_node)

    # Stage B: relaxed match (lemma/theorem equivalence only) — used only if strict match fails.
    proof_node, matched_sig = _find_last_matching_proof_node(allow_lemma_theorem_equivalence=True)
    if proof_node:
        logger.debug(
            "Signature match used lemma/theorem equivalence. target=%r matched=%r",
            str(target_signature),
            str(matched_sig),
        )
        return _extract_tactics_from_proof_node(proof_node)

    # Stage C: relaxed match for whitespace immediately before closing delimiters.
    # This is intentionally narrow: we preserve strict matching unless no declaration matches.
    def _relaxed_closing_delim_key(sig: str) -> str:
        return _remove_space_before_closing_delims(_canonicalize_lemma_theorem(sig))

    target_key = _relaxed_closing_delim_key(target_signature)
    matching_proof_node: dict | None = None
    matched_sig = None

    for cmd in commands:
        if not isinstance(cmd, dict):
            continue

        # Handle Lean.Parser.Command.declaration wrapper
        target_node = cmd
        kind = cmd.get("kind", "")
        if kind == "Lean.Parser.Command.declaration":
            args = cmd.get("args", [])
            for arg in args:
                if isinstance(arg, dict) and __is_theorem_or_lemma_kind(arg.get("kind")):
                    target_node = arg
                    break

        decl_type = target_node.get("type") or cmd.get("type")
        if decl_type and _relaxed_closing_delim_key(decl_type) == target_key:
            proof_node = _find_proof_body_node_structurally(target_node)
            if proof_node:
                matching_proof_node = proof_node
                matched_sig = str(decl_type)
                continue

        reconstructed_sig = _reconstruct_signature_from_decl_node(target_node, skip_comments=True)
        if _relaxed_closing_delim_key(reconstructed_sig) == target_key:
            proof_node = _find_proof_body_node_structurally(target_node)
            if proof_node:
                matching_proof_node = proof_node
                matched_sig = reconstructed_sig

    if matching_proof_node:
        logger.debug(
            "Signature match used closing-delimiter whitespace normalization. target=%r matched=%r",
            str(target_signature),
            str(matched_sig),
        )
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

    # 2. Fallback: Proof might be a (recursive) direct child of the declaration's args (common in dummy ASTs)
    def _recursively_find_proof_node(args: dict) -> dict | None:
        seen_assign = False
        for arg in args:
            if not isinstance(arg, dict):
                continue
            if arg.get("kind", "") == "group" and arg.get("args", []):
                return _find_proof_body_node_structurally(arg)
            if arg.get("val") == ":=":
                seen_assign = True
                continue
            if seen_assign:
                proof_node = _check_if_proof_node(arg)
                if proof_node:
                    return proof_node
        return None

    return _recursively_find_proof_node(args)


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
            # IMPORTANT: Do NOT `.strip()` the tactic body.
            #
            # `.strip()` removes leading spaces from only the *first* line (subsequent lines keep
            # their indentation since they're preceded by `\n`). For proofs with multiple aligned
            # top-level tactics (e.g. multiple `have ... := by` blocks), that breaks relative
            # indentation and can later make inlining/reconstruction fail with
            # "All indentation strategies failed".
            #
            # When Kimina doesn't include indentation spaces in the `by` token's trailing
            # (`next_line_indent == ""`), stripping is especially harmful: we would drop the first
            # line's indentation and incorrectly nest later tactics.
            tactics = _ast_to_code(by_args[1])
            tactics = tactics.lstrip("\n").rstrip()
            if next_line_indent:
                # Preserve existing relative structure by only normalizing the *first line*'s
                # indentation, then prefixing the indentation extracted from `by`.
                tactics = tactics.lstrip(" \t")
                return next_line_indent + tactics
            return tactics
    elif kind == "Lean.Parser.Tactic.tacticSeq":
        return _ast_to_code(proof_node).lstrip("\n").rstrip()

    return _ast_to_code(proof_node).lstrip("\n").rstrip()


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
