# DOI release procedure

This repository is prepared for GitHub-to-Zenodo archiving. Publishing is deliberately a separate final action because it makes the package public and creates a permanent scholarly record.

## Final pre-release checks

1. Replace any remaining manuscript author/contact placeholders outside this repository.
2. Run `python run_all.py`, the unit tests, and `python tools/validate_repository.py` in a clean environment.
3. Review `LICENSE`, `LICENSE-DATA.md`, `NOTICE`, `docs/THIRD_PARTY_DATA.md`, and the privacy scan report.
4. Create the public GitHub repository and push the verified commit.
5. In Zenodo, connect the GitHub account, synchronize repositories, and enable this repository.
6. Create a GitHub release tagged `v1.0.0`. Zenodo archives enabled public repositories when a GitHub release is created and then assigns a version DOI.
7. Verify the Zenodo record title, creator, ORCID, version, license, files, and description before citing it.
8. Insert the assigned version DOI into the manuscript's Data and Code Availability statement. Add the concept DOI to the repository README later if desired.

Zenodo states that when both `.zenodo.json` and `CITATION.cff` exist, `.zenodo.json` supplies the Zenodo metadata. The two files in this repository are therefore kept consistent.

Official guidance:

- Zenodo, enabling a GitHub repository: <https://help.zenodo.org/docs/github/enable-repository/>
- Zenodo, describing software: <https://help.zenodo.org/docs/github/describe-software/>
- Zenodo, archiving a GitHub release: <https://help.zenodo.org/docs/github/archive-software/github-upload/>
- GitHub, referencing and citing content: <https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content>

Do not publish a release until the author has reviewed the public file list and third-party reuse terms.
