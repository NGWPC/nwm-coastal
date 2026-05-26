"""Strip ANSI escape codes and absolute repo paths from notebook cell outputs."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHFABCDEFGHJKSTsu]")


def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def clean_text(text: str, abs_prefix: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace(abs_prefix, "")
    return text


def clean_notebook(path: Path, abs_prefix: str) -> None:
    nb = json.loads(path.read_text())
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            for key in ("text", "traceback"):
                if key not in output:
                    continue
                val = output[key]
                if isinstance(val, list):
                    output[key] = [clean_text(line, abs_prefix) for line in val]
                else:
                    output[key] = clean_text(val, abs_prefix)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    root = repo_root()
    abs_prefix = str(root) + "/"

    if len(sys.argv) > 1:
        notebooks = [Path(f) for f in sys.argv[1:] if f.endswith(".ipynb")]
    else:
        notebooks = sorted((root / "docs/examples/notebooks").glob("*.ipynb"))

    for nb_path in notebooks:
        clean_notebook(nb_path, abs_prefix)


if __name__ == "__main__":
    main()
