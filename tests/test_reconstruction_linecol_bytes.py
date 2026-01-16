from __future__ import annotations

from goedels_poetry.parsers.ast import AST


def _make_ast(source: str) -> AST:
    return AST({"kind": "root", "args": []}, source_text=source, body_start=0)


def test_line_col_ascii_byte_mapping() -> None:
    source = "ab\ncde\n"
    ast = _make_ast(source)
    offset = ast.line_col_to_byte_offset(line=2, column=1, column_is_byte=False, line_base=1)
    expected = len(b"ab\n") + len(b"c")
    assert offset == expected


def test_line_col_unicode_multibyte_mapping() -> None:
    source = "αβ\nγ\n"  # noqa: RUF001
    ast = _make_ast(source)
    offset = ast.line_col_to_byte_offset(line=1, column=1, column_is_byte=False, line_base=1)
    expected = len("α".encode())  # noqa: RUF001
    assert offset == expected


def test_line_col_combining_chars_mapping() -> None:
    source = "e\u0301x\n"
    ast = _make_ast(source)
    offset = ast.line_col_to_byte_offset(line=1, column=1, column_is_byte=False, line_base=1)
    expected = len(b"e")
    assert offset == expected
