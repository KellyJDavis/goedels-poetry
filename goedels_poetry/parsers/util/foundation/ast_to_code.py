"""AST to Lean text conversion."""

from typing import Any


def _ast_to_code(node: Any) -> str:  # noqa: C901
    if isinstance(node, dict):
        kind = node.get("kind", "")
        # Handle custom containers
        if kind == "__value_container":
            # Just serialize the args directly
            return "".join(_ast_to_code(arg) for arg in node.get("args", []))
        if kind == "__type_container":
            # Just serialize the args directly
            return "".join(_ast_to_code(arg) for arg in node.get("args", []))
        if kind == "__equality_expr":
            # Serialize as "var = value"
            return "".join(_ast_to_code(arg) for arg in node.get("args", []))

        def _atom_text(d: dict[str, Any]) -> str:
            """
            Convert a Kimina AST "atom"/"ident" node into surface text.

            Kimina uses a sentinel `val == "[anonymous]"` (often with `rawVal == ""`) to represent
            "no token here" / anonymous binders in the original source. Emitting it literally
            produces invalid Lean (e.g. `have [anonymous]: ...`), so we must suppress it.
            """
            # Prefer rawVal when it is present and non-empty; it is closest to the original token.
            raw = d.get("rawVal")
            if isinstance(raw, str) and raw != "":
                return raw

            v_any: Any = d.get("val", "")
            v = v_any if isinstance(v_any, str) else str(v_any)
            if v == "[anonymous]":
                return ""
            return v

        parts = []
        if "val" in node:
            info = node.get("info", {}) or {}
            leading = info.get("leading", "")
            trailing = info.get("trailing", "")
            parts.append(f"{leading}{_atom_text(node)}{trailing}")
        # prefer 'args' order first (parser uses args for ordered tokens)
        for arg in node.get("args", []):
            parts.append(_ast_to_code(arg))
        # then traverse other fields conservatively
        for k, v in node.items():
            if k in {"args", "val", "info", "kind"}:
                continue
            parts.append(_ast_to_code(v))
        return "".join(parts)
    elif isinstance(node, list):
        return "".join(_ast_to_code(x) for x in node)
    else:
        return ""
