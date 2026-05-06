"""Expression trees and genetic programming operators for proxy formula search."""

import copy
import json
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


# --- Node types ---

class Node(ABC):
    """Abstract base class for expression tree nodes."""

    @abstractmethod
    def evaluate(self, stats: Dict[str, float]) -> float:
        ...

    @abstractmethod
    def depth(self) -> int:
        ...

    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def copy(self) -> "Node":
        ...

    @abstractmethod
    def to_str(self) -> str:
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        ...

    def __repr__(self) -> str:
        return self.to_str()


class Terminal(Node):
    """Leaf node referencing a named statistic."""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, stats: Dict[str, float]) -> float:
        return stats.get(self.name, 0.0)

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 1

    def copy(self) -> "Terminal":
        return Terminal(self.name)

    def to_str(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        return {"type": "terminal", "name": self.name}


class Constant(Node):
    """Leaf node with an ephemeral random constant."""

    def __init__(self, value: float):
        self.value = value

    def evaluate(self, stats: Dict[str, float]) -> float:
        return self.value

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 1

    def copy(self) -> "Constant":
        return Constant(self.value)

    def to_str(self) -> str:
        return f"{self.value:.4f}"

    def to_dict(self) -> dict:
        return {"type": "constant", "value": self.value}


# --- Protected math operations ---

def _protected_div(x: float, y: float) -> float:
    sign_y = 1.0 if y >= 0 else -1.0
    return x / (y + sign_y * 1e-8)


def _protected_log(x: float) -> float:
    return math.log(abs(x) + 1e-8)


def _protected_sqrt(x: float) -> float:
    return math.sqrt(abs(x) + 1e-8)


def _protected_inv(x: float) -> float:
    sign_x = 1.0 if x >= 0 else -1.0
    return 1.0 / (x + sign_x * 1e-8)


UNARY_OPS = {
    "log": _protected_log,
    "abs": abs,
    "sqrt": _protected_sqrt,
    "neg": lambda x: -x,
    "inv": _protected_inv,
}

BINARY_OPS = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": _protected_div,
}


class UnaryOp(Node):
    """Unary operator node."""

    def __init__(self, op: str, child: Node):
        self.op = op
        self.child = child
        self._fn = UNARY_OPS[op]

    def evaluate(self, stats: Dict[str, float]) -> float:
        return self._fn(self.child.evaluate(stats))

    def depth(self) -> int:
        return 1 + self.child.depth()

    def size(self) -> int:
        return 1 + self.child.size()

    def copy(self) -> "UnaryOp":
        return UnaryOp(self.op, self.child.copy())

    def to_str(self) -> str:
        return f"{self.op}({self.child.to_str()})"

    def to_dict(self) -> dict:
        return {"type": "unary", "op": self.op, "child": self.child.to_dict()}


class BinaryOp(Node):
    """Binary operator node."""

    def __init__(self, op: str, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right
        self._fn = BINARY_OPS[op]

    def evaluate(self, stats: Dict[str, float]) -> float:
        return self._fn(self.left.evaluate(stats), self.right.evaluate(stats))

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def copy(self) -> "BinaryOp":
        return BinaryOp(self.op, self.left.copy(), self.right.copy())

    def to_str(self) -> str:
        return f"({self.left.to_str()} {self.op} {self.right.to_str()})"

    def to_dict(self) -> dict:
        return {
            "type": "binary",
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


# --- Statistic names (GP terminal set) ---

STAT_NAMES = [
    "act_variance", "act_kurtosis", "act_sparsity",
    "feature_rank", "feature_norm_ratio",
    "layer_cka_mean", "layer_cka_std",
    "grad_norm", "grad_snr", "grad_conflict",
    "param_count_log",
    "dl_score", "pl_score", "ta_score",
]


# --- Tree generation ---

def random_tree(max_depth: int, rng: np.random.Generator, method: str = "grow") -> Node:
    """Generate a random expression tree.

    Args:
        max_depth: Maximum tree depth.
        rng: Numpy random generator.
        method: "grow" (mixed) or "full" (always expand to max_depth).

    Returns:
        Root node of the generated tree.
    """
    if max_depth <= 0:
        return _random_leaf(rng)

    # Terminal probability increases with lower remaining depth for "grow"
    if method == "grow":
        terminal_prob = 1.0 / (max_depth + 1)
        if rng.random() < terminal_prob:
            return _random_leaf(rng)

    # Choose operator type
    if rng.random() < 0.5:
        # Unary
        op = rng.choice(list(UNARY_OPS.keys()))
        child = random_tree(max_depth - 1, rng, method)
        return UnaryOp(op, child)
    else:
        # Binary
        op = rng.choice(list(BINARY_OPS.keys()))
        left = random_tree(max_depth - 1, rng, method)
        right = random_tree(max_depth - 1, rng, method)
        return BinaryOp(op, left, right)


def _random_leaf(rng: np.random.Generator) -> Node:
    """Generate a random leaf: 80% named terminal, 20% constant."""
    if rng.random() < 0.2:
        value = rng.uniform(-2.0, 2.0)
        return Constant(round(float(value), 4))
    else:
        name = rng.choice(STAT_NAMES)
        return Terminal(name)


# --- GP operators ---

def _all_nodes(tree: Node) -> List[Tuple[Node, Optional[Node], Optional[str]]]:
    """Collect all (node, parent, child_key) triples via DFS."""
    result = [(tree, None, None)]
    stack = [(tree, None, None)]
    while stack:
        node, parent, key = stack.pop()
        if isinstance(node, UnaryOp):
            result.append((node.child, node, "child"))
            stack.append((node.child, node, "child"))
        elif isinstance(node, BinaryOp):
            result.append((node.left, node, "left"))
            result.append((node.right, node, "right"))
            stack.append((node.left, node, "left"))
            stack.append((node.right, node, "right"))
    return result


def point_mutation(tree: Node, rng: np.random.Generator, max_depth: int = 5) -> Node:
    """Replace a random subtree with a new random subtree."""
    tree = tree.copy()
    nodes = _all_nodes(tree)

    idx = rng.integers(0, len(nodes))
    target, parent, key = nodes[idx]

    # Compute remaining depth budget
    if parent is None:
        # Replacing root
        remaining_depth = max_depth
    else:
        # Approximate: use max_depth minus current position depth
        remaining_depth = max(1, max_depth - (tree.depth() - target.depth()))

    new_subtree = random_tree(min(remaining_depth, 3), rng, method="grow")

    if parent is None:
        return new_subtree
    setattr(parent, key, new_subtree)

    # Reject if bloated
    if tree.depth() > max_depth:
        return tree.copy()  # Return unmodified copy
    return tree


def subtree_crossover(
    p1: Node, p2: Node, rng: np.random.Generator, max_depth: int = 5
) -> Node:
    """Swap random subtrees between two parents. Returns one offspring."""
    child = p1.copy()
    donor = p2.copy()

    child_nodes = _all_nodes(child)
    donor_nodes = _all_nodes(donor)

    # Pick crossover points
    c_idx = rng.integers(0, len(child_nodes))
    d_idx = rng.integers(0, len(donor_nodes))

    _, c_parent, c_key = child_nodes[c_idx]
    d_node, _, _ = donor_nodes[d_idx]

    if c_parent is None:
        # Replace root
        child = d_node.copy()
    else:
        setattr(c_parent, c_key, d_node.copy())

    # Reject if bloated
    if child.depth() > max_depth:
        return p1.copy()
    return child


# --- Evaluation ---

def evaluate_proxy(tree: Node, stats: Dict[str, float]) -> float:
    """Evaluate proxy tree on a stats dict. Returns 0.0 on any error."""
    try:
        val = tree.evaluate(stats)
        if not math.isfinite(val):
            return 0.0
        return max(-1e6, min(1e6, val))
    except Exception:
        return 0.0


# --- Serialization ---

def serialize_tree(tree: Node) -> dict:
    """Convert tree to JSON-serializable dict."""
    return tree.to_dict()


def deserialize_tree(d: dict) -> Node:
    """Reconstruct tree from dict."""
    node_type = d["type"]
    if node_type == "terminal":
        return Terminal(d["name"])
    elif node_type == "constant":
        return Constant(d["value"])
    elif node_type == "unary":
        child = deserialize_tree(d["child"])
        return UnaryOp(d["op"], child)
    elif node_type == "binary":
        left = deserialize_tree(d["left"])
        right = deserialize_tree(d["right"])
        return BinaryOp(d["op"], left, right)
    raise ValueError(f"Unknown node type: {node_type}")
