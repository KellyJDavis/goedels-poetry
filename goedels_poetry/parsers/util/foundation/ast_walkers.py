"""Generic AST walkers and search functions."""

from collections.abc import Callable
from typing import Any

from .constants import Node


def __find_first(node: Node, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if predicate(node):
            return node
        for v in node.values():
            res = __find_first(v, predicate)
            if res is not None:
                return res
    elif isinstance(node, list):
        for it in node:
            res = __find_first(it, predicate)
            if res is not None:
                return res
    return None


def __find_all(
    node: Node, predicate: Callable[[dict[str, Any]], bool], acc: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if acc is None:
        acc = []
    if isinstance(node, dict):
        if predicate(node):
            acc.append(node)
        for v in node.values():
            __find_all(v, predicate, acc)
    elif isinstance(node, list):
        for it in node:
            __find_all(it, predicate, acc)
    return acc
