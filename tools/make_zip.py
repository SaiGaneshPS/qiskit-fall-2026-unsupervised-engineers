"""Build the submission ZIP: <team number>_<team name>.zip, next to the project folder.

Reads the team number and name from submit_info.md, refuses to run while they are still TODO (use --dry-run to
check the contents), and includes everything the rule book asks for: submit_info.md, slides, notebook, source code,
requirements.txt, data inputs, results and figures.
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["submit_info.md", "README.md", "notebook.ipynb", "requirements.txt", ".gitignore",
           "gridrisk", "experiments", "tools", "data", "results", "figures", "slides"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def team_fields() -> tuple[str, str]:
    s = (ROOT / "submit_info.md").read_text(encoding="utf-8")
    num = re.search(r"\*\*Team number:\*\*\s*(.+)", s).group(1).strip()
    name = re.search(r"\*\*Team name:\*\*\s*(.+)", s).group(1).strip()
    return num, name


def files():
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (SKIP_PARTS & set(f.parts)):
                    yield f


def main(dry_run: bool):
    num, name = team_fields()
    slug = re.sub(r"[^a-z0-9]+", "", name.lower())
    fl = list(files())
    size = sum(f.stat().st_size for f in fl) / 1e6
    print(f"{len(fl)} files, {size:.1f} MB uncompressed")
    if not any(f.parent.name == "slides" for f in fl):
        print("WARNING: no slides found in slides/ (the rule book requires them in the ZIP)")
    if dry_run:
        for f in fl:
            print("  ", f.relative_to(ROOT))
        return
    if "TODO" in (num + name) or not slug:
        raise SystemExit("Fill in the team number and team name in submit_info.md first.")
    out = ROOT.parent / f"{num}_{slug}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in fl:
            z.write(f, Path(ROOT.name) / f.relative_to(ROOT))
    print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
