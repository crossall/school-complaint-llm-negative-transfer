from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) != 2 or not re.fullmatch(r"10\.\d{4,9}/\S+", sys.argv[1]):
        raise SystemExit("usage: python tools/set_doi.py 10.xxxx/zenodo.xxxxxxx")
    doi = sys.argv[1]
    citation = ROOT / "CITATION.cff"
    text = citation.read_text(encoding="utf-8")
    if re.search(r"^doi:", text, flags=re.MULTILINE):
        text = re.sub(r'^doi:.*$', f'doi: "{doi}"', text, flags=re.MULTILINE)
    else:
        text = text.replace("version: 1.0.0\n", f'version: 1.0.0\ndoi: "{doi}"\n')
    citation.write_text(text, encoding="utf-8")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("DOI to be assigned.", f"https://doi.org/{doi}.")
    text = text.replace("DOI will be inserted after the Zenodo release.", f"DOI: https://doi.org/{doi}.")
    readme.write_text(text, encoding="utf-8")
    print(f"Inserted {doi} into CITATION.cff and README.md. Rebuild MANIFEST.sha256 before release.")


if __name__ == "__main__":
    main()
