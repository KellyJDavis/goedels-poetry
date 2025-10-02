from __future__ import annotations

from typing import Protocol, runtime_checkable

# TODO: Document these classes


@runtime_checkable
class TreeNode(Protocol):
    @property
    def parent(self) -> TreeNode | None: ...

    @property
    def depth(self) -> int: ...


@runtime_checkable
class InternalTreeNode(TreeNode, Protocol):
    @property
    def children(self) -> list[TreeNode]: ...
