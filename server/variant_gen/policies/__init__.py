"""Language/course-specific policy hooks for variant generation.

Keep generator invariants generic; put language- or course-specific rules here.
"""

from .python_intro import PythonIntroPolicy


def get_policy(language: str):
    if language == "python":
        return PythonIntroPolicy()
    return None

