from __future__ import annotations

from typing import Protocol, cast, runtime_checkable


@runtime_checkable
class TreeNode(Protocol):
    """
    Protocol supposed by all tree nodes.
    """

    @property
    def id(self) -> str:
        """
        The id of this tree node.

        Returns
        -------
        str
            The id of this tree node.
        """
        ...

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
    def children(self) -> dict[str, TreeNode]:
        """
        The children of this internal tree node keyed by their id.

        Returns
        -------
        dict[str, TreeNode]
            The children of this internal tree node keyed by their id.
        """
        ...


def add_child(parent: InternalTreeNode, child: TreeNode) -> None:
    """
    Add a child to an internal tree node. The child is stored under its id.

    Parameters
    ----------
    parent : InternalTreeNode
        The parent node (must have a mutable children dict).
    child : TreeNode
        The child node to add. Must have an "id" key (e.g. TypedDict).
    """
    cast(dict, parent)["children"][cast(dict, child)["id"]] = child


def remove_child(parent: InternalTreeNode, child: TreeNode) -> None:
    """
    Remove a child from an internal tree node by the child's id.

    Parameters
    ----------
    parent : InternalTreeNode
        The parent node (must have a mutable children dict).
    child : TreeNode
        The child node to remove. Must have an "id" key.
    """
    del cast(dict, parent)["children"][cast(dict, child)["id"]]
