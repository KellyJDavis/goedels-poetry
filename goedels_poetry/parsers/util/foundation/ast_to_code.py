"""AST to Lean text conversion."""

from typing import Any


def _ast_to_code(node: Any) -> str:
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

        parts = []
        if "val" in node:
            info = node.get("info", {}) or {}
            leading = info.get("leading", "")
            trailing = info.get("trailing", "")
            parts.append(f"{leading}{node['val']}{trailing}")
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
