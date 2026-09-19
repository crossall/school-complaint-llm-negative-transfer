from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"
IGNORED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}


def is_ignored(path: Path) -> bool:
    return bool(IGNORED_DIRS.intersection(path.relative_to(ROOT).parts))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file() or is_ignored(path) or path == OUTPUT:
            continue
        rows.append(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}")
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(rows)} hashes to {OUTPUT.name}")


if __name__ == "__main__":
    main()
