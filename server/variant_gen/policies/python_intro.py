import ast
import re
from typing import Optional


class PythonIntroPolicy:
    """Python-intro specific invariants and autofixes."""

    LIST_METHOD_NAMES = frozenset(
        {
            "append",
            "clear",
            "copy",
            "count",
            "extend",
            "index",
            "insert",
            "pop",
            "remove",
            "reverse",
            "sort",
        }
    )

    @staticmethod
    def original_suggests_list_method_mcq(original_text: str) -> bool:
        t = (original_text or "").lower()
        if not re.search(r"(?:^|\n|\s)(?:[A-E]|[1-5])[\.\)]\s+\w+", original_text or ""):
            return False
        # "method reverseList(head)" on a linked list is not a built-in list.append-style MCQ.
        if re.search(r"linked[\s-]*list|singly[\s-]*linked", t):
            return False
        compact = re.sub(r"\s+", "", t)
        if "reverselist" in compact:
            return False
        return (
            "list method" in t
            or "elements to a list" in t
            or ("to a list" in t and "method" in t)
            or (
                "method" in t
                and "list" in t
                and "linked" not in t
            )
        )

    @staticmethod
    def stem_asks_for_non_method(stem_lower: str) -> bool:
        s = stem_lower or ""
        if "method" not in s or not re.search(r"\bnot\b", s):
            return False
        if re.search(r"\bnot\s+a\s+list\s+method\b", s):
            return True
        if re.search(r"\bnot\b.?\s*(one\s+of\s+)?the\s+following", s):
            return True
        if re.search(r"which\b.*\bnot\b.*\bmethod\b", s):
            return True
        return False

    def list_method_mcq_autofix(self, variant: dict, original_text: str, normalize_answer, mcq_correct_option_label) -> bool:
        """If exactly one option is (non-)method, set correct_answer accordingly."""
        if not self.original_suggests_list_method_mcq(original_text):
            return False
        opts = variant.get("options")
        if not opts or not isinstance(opts, dict):
            return False
        vt = (variant.get("variant_text") or "").lower()
        old = str(variant.get("correct_answer", "")).strip()

        neg = self.stem_asks_for_non_method(vt)
        if neg:
            bad = [
                k
                for k, v in opts.items()
                if str(v).strip().lower() not in self.LIST_METHOD_NAMES
            ]
            if len(bad) != 1:
                return False
            pick = str(bad[0]).strip()
        else:
            good = [
                k
                for k, v in opts.items()
                if str(v).strip().lower() in self.LIST_METHOD_NAMES
            ]
            if len(good) != 1:
                return False
            pick = str(good[0]).strip()

        # If options are 1–5 but answer is A–E, normalize.
        variant["correct_answer"] = normalize_answer(pick)
        return normalize_answer(old) != normalize_answer(variant["correct_answer"])

    def validate_list_method_mcq(self, variant: dict, original_text: str, vt_lower: str, mcq_correct_option_label) -> Optional[str]:
        if not self.original_suggests_list_method_mcq(original_text):
            return None
        opts = variant.get("options")
        if not opts or not isinstance(opts, dict):
            return None
        real = [k for k, v in opts.items() if str(v).strip().lower() in self.LIST_METHOD_NAMES]
        fake = [k for k, v in opts.items() if str(v).strip().lower() not in self.LIST_METHOD_NAMES]
        if not real:
            return "list-method MCQ: at least one option must be a real Python list method"
        neg = self.stem_asks_for_non_method(vt_lower)
        if neg and len(fake) != 1:
            return f"list-method MCQ: expected exactly 1 non-method option, got {len(fake)}"
        if not neg and len(real) != 1:
            return f"list-method MCQ: expected exactly 1 real list method option, got {len(real)}"

        lab, opt_txt = mcq_correct_option_label(variant.get("correct_answer"), opts)
        if lab and opt_txt is not None:
            if neg and opt_txt in self.LIST_METHOD_NAMES:
                return "stem asks which is NOT a list method but correct_answer picks a real method"
            if not neg and opt_txt not in self.LIST_METHOD_NAMES:
                return "stem asks for a list method but correct_answer is not a list method"
        return None

    @staticmethod
    def extract_python_for_parse(s: str) -> str:
        if not s:
            return ""
        m = re.search(r"```(?:python)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return s.strip()

    def answer_parseable(self, answer: str) -> bool:
        code = self.extract_python_for_parse(answer)
        if "def " not in code and "class " not in code:
            return True
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def answer_has_mutable_defaults(self, answer: str) -> bool:
        code = self.extract_python_for_parse(answer)
        return bool(
            re.search(r"def\s+\w+\s*\([^)]*=\s*\[\s*\]", code)
            or re.search(r"def\s+\w+\s*\([^)]*=\s*\{\s*\}", code)
        )

    @staticmethod
    def original_shows_mutable_defaults(original_text: str) -> bool:
        o = original_text or ""
        return bool(re.search(r"=\s*\[\s*\]", o) or re.search(r"=\s*\{\s*\}", o))

