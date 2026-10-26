# Contributing

Use Python 3.11 or newer and a virtual environment. Install `requirements-dev.txt`, run `python -m unittest discover -s tests -v`, and compile the source with `python -m compileall -q main.py config.py agent gui` before opening a pull request.

Keep changes focused and update documentation when behavior changes. For bugs, include the command, Python version, dependency versions, and a small synthetic reproducer. Remove API keys, private data, local paths, and sensitive run logs from reports.

Do not commit `.env`, generated outputs, checkpoints, virtual environments, or private datasets. Add meaningful offline regression coverage for behavior fixes. Live provider checks are optional and must use your own key; never require a secret in CI.

Scientific changes should include an explicit formulation, units, assumptions, baseline, and reproducible numerical evidence. Model-generated peer review alone is insufficient validation.

No license has been selected. Discuss reuse and contribution terms with the owner before contributing substantial third-party code or datasets.
