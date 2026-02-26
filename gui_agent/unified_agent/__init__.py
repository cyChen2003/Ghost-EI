import sys
from pathlib import Path
from typing import Optional


def _ensure_repo_root_on_path() -> None:
    """Add the project root (containing MobileAgent) to sys.path if needed."""
    repo_root: Optional[Path] = None
    for parent in Path(__file__).resolve().parents:
        mobile_agent_dir = parent / "MobileAgent"
        if mobile_agent_dir.is_dir():
            repo_root = parent
            break
    if not repo_root:
        return

    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_repo_root_on_path()

from .actions import MobileAction
from .agents import OSAtlasAgent, UITarsAgent, TARS15Agent
from .runner import evaluate_dataset, run_all_tests

__all__ = [
    "MobileAction",
    "OSAtlasAgent",
    "UITarsAgent",
    "TARS15Agent",
    "run_all_tests",
    "evaluate_dataset",
]
