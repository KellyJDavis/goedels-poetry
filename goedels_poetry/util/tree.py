from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TreeNode(Protocol):
    """
    Protocol supposed by all tree nodes.
    """

    @property
    def parent(self) -> TreeNode | None:
        """
        The parent of this tree node.

        Returns
        -------
        TreeNode
            The parent of this tree node.
        """
        ...

    @property
    def depth(self) -> int:
        """
        Depth of this TreeNode from the root; root has depth 0.

        Returns
        -------
        int
            Depth of this TreeNode from the root; root has depth 0.
        """
        ...


@runtime_checkable
class InternalTreeNode(TreeNode, Protocol):
    """
    Protocol supposed by all internal tree nodes.
    """

    @property
    def children(self) -> list[TreeNode]:
        """
        The children of this internal tree node.

        Returns
        -------
        list[TreeNode]
            The children of this internal tree node.
        """
        ...
