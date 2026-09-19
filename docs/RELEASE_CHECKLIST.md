# Public release checklist

- [ ] `python run_all.py` succeeds in a clean Python 3.11 environment.
- [ ] `python -m unittest discover -s tests -v` passes.
- [ ] `python tools/validate_repository.py` reports PASS.
- [ ] `MANIFEST.sha256` matches all packaged files.
- [ ] No AI Hub raw/processed training data are present.
- [ ] No Qwen or QLoRA weights/checkpoints are present.
- [ ] No full 3,716-chunk official-document corpus is present.
- [ ] No direct rater identifiers or local absolute paths are present.
- [ ] Author has reviewed benchmark, model outputs, and licenses for public release.
- [ ] GitHub repository is public and Zenodo integration is enabled.
- [ ] GitHub release `v1.0.0` is created from the verified commit.
- [ ] Zenodo version DOI is verified and inserted into the manuscript.
