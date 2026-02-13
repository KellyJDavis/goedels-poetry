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
            v_any: Any = d.get("val", "")
            v_str = v_any if isinstance(v_any, str) else None

            # Always suppress Kimina's anonymous sentinel.
            if v_str == "[anonymous]":
                return ""

            raw = d.get("rawVal")
            raw_str = raw if isinstance(raw, str) else None

            # If rawVal exists, use it only when it matches the current val.
            # This keeps output faithful to original tokenization, but remains rewrite-aware:
            # variable renaming mutates `val`, while Kimina's `rawVal` stays as originally parsed.
            if raw_str:
                if v_str is not None and v_str != raw_str:
                    return v_str
                return raw_str

            # Fallbacks: prefer val when it's a string, otherwise stringify.
            if v_str is not None:
                return v_str
            return str(v_any)

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
