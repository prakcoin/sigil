from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from watchfiles import watch


def start(
    project_root: Path,
    on_change: Callable[[set[Path]], None],
) -> threading.Event:
    """Start a background file watcher thread.

    Calls on_change with the set of changed paths whenever a .py,
    SKILL.md, or sigil.yaml file is modified.

    Returns the stop event — set it to shut the watcher down cleanly.
    """
    stop_event = threading.Event()

    def _run() -> None:
        for changes in watch(str(project_root), stop_event=stop_event):
            paths = {
                Path(p)
                for _, p in changes
                if p.endswith(".py") or p.endswith("SKILL.md") or p.endswith("sigil.yaml")
            }
            if paths:
                on_change(paths)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return stop_event
