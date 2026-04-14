"""Environment, paths, and model constants for variant_gen."""

import os
from pathlib import Path

from dotenv import load_dotenv

from .exam_tests_questions import EXAM_TESTS_DIR

_SERVER_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_SERVER_DIR / ".env")
load_dotenv()

BASE_DIR = _SERVER_DIR.parent
DB_PATH = EXAM_TESTS_DIR

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default when OPENROUTER_MODEL is unset (OpenRouter slug).
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


def resolved_openrouter_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "").strip() or DEFAULT_OPENROUTER_MODEL


def openrouter_vision_enabled() -> bool:
    """Multimodal requests allowed (crop + text in one call)."""
    return os.getenv("OPENROUTER_VISION", os.getenv("OPENROUTER_SEND_IMAGES", "1")).lower() not in (
        "0",
        "false",
        "no",
    )


DEBUG = False
MAX_RETRIES = 3
BASE_TEMPERATURE = 0.4
SIMILARITY_THRESHOLD = 0.6
SIMILARITY_THRESHOLD_EXPLANATION = 0.72

_ALGORITHM_PATTERNS = [
    ("hash map lookup", ["hash", "dictionary", "dict", "lookup", "two sum", "key", "mapping"]),
    ("graph traversal", ["graph", "bfs", "dfs", "breadth", "depth", "adjacen", "vertex", "edge", "shortest path", "dijkstra", "spanning tree", "kruskal"]),
    ("dynamic programming", ["dynamic programming", "memoiz", "tabulation", "subproblem", "overlapping", "optimal substructure", "longest common", "knapsack"]),
    ("sorting / ordering", ["sort", "order", "rank", "arrange", "ascending", "descending", "merge sort", "quicksort", "bubble"]),
    ("binary search", ["binary search", "bisect", "sorted array", "log n", "divide and conquer"]),
    ("recursion", ["recurs", "base case", "recursive"]),
    ("tree traversal", ["tree", "preorder", "inorder", "postorder", "binary tree", "bst", "traversal"]),
    ("linked list", ["linked list", "singly linked", "head", "node", "next pointer", "reverse"]),
    ("stack / queue", ["stack", "queue", "push", "pop", "peek", "fifo", "lifo"]),
    ("set operations", ["set", "intersection", "union", "difference", "common elements"]),
    ("string manipulation", ["string", "substring", "reverse", "palindrome", "anagram", "character"]),
    ("array search", ["array", "list", "find", "search", "index", "element"]),
    ("greedy", ["greedy", "optimal", "locally optimal"]),
    ("class design / OOP", ["class", "inherit", "object", "method", "__init__", "__str__", "attribute"]),
]
