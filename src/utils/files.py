from __future__ import annotations

from pathlib import Path
import re


def next_available_index(directory: Path, prefix: str = "img_", suffix: str = ".png", width: int = 4) -> int:
    """Return the next available numeric index based on existing files.

    Scans for files named like: f"{prefix}<N:{width}d>{suffix}" and returns max(N)+1.
    Starts at 1 if none exist.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{{width}}}){re.escape(suffix)}$")
    max_idx = 0
    if directory.exists():
        for p in directory.iterdir():
            if p.is_file():
                m = pattern.match(p.name)
                if m:
                    try:
                        idx = int(m.group(1))
                        if idx > max_idx:
                            max_idx = idx
                    except ValueError:
                        continue
    return max_idx + 1


__all__ = ["next_available_index"]
