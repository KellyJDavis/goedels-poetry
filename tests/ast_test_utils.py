from __future__ import annotations

from typing import Any

from goedels_poetry.parsers.ast import AST


def _char_to_byte_index(text: str, idx: int) -> int:
    return len(text[:idx].encode("utf-8"))


def _make_token(val: str, start: int, end: int, *, source_text: str) -> dict[str, Any]:
    start_b = _char_to_byte_index(source_text, start)
    end_b = _char_to_byte_index(source_text, end)
    return {"val": val, "info": {"pos": [start_b, end_b], "leading": "", "trailing": ""}}


def build_simple_ast(
    source_text: str,
    *,
    proof_body_span: tuple[int, int] | None = None,
    sorry_spans: list[tuple[int, int]] | None = None,
) -> AST:
    by_start = source_text.index("by")
    by_end = by_start + len("by")
    if proof_body_span is None:
        proof_body_span = (by_end, len(source_text))

    tokens: list[dict[str, Any]] = []
    if proof_body_span[1] > proof_body_span[0]:
        body_text = source_text[proof_body_span[0] : proof_body_span[1]]
        tokens.append(_make_token(body_text, proof_body_span[0], proof_body_span[1], source_text=source_text))

    for start, end in sorry_spans or []:
        tokens.append({
            "kind": "Lean.Parser.Tactic.tacticSorry",
            "args": [_make_token("sorry", start, end, source_text=source_text)],
        })

    def token_start(tok: dict[str, Any]) -> int:
        if "kind" in tok:
            info = tok["args"][0]["info"]
            return int(info["pos"][0])
        return int(tok["info"]["pos"][0])

    tokens.sort(key=token_start)
    tactic_seq = {"kind": "Lean.Parser.Tactic.tacticSeq", "args": tokens}
    by_token = _make_token("by", by_start, by_end, source_text=source_text)
    by_tactic = {"kind": "Lean.Parser.Term.byTactic", "args": [by_token, tactic_seq]}
    ast = {"kind": "Lean.Parser.Command.theorem", "args": [by_tactic]}
    return AST(ast, source_text=source_text, body_start=0)


def find_sorry_spans(source_text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    idx = 0
    while True:
        idx = source_text.find("sorry", idx)
        if idx == -1:
            break
        before = source_text[idx - 1] if idx > 0 else " "
        after_idx = idx + len("sorry")
        after = source_text[after_idx] if after_idx < len(source_text) else " "
        if before.isspace() and after.isspace():
            spans.append((idx, after_idx))
        idx = after_idx
    return spans
