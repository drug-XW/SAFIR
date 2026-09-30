# Upload to GitHub

1. Unzip `SAFIR_GitHub_complete.zip` locally.
2. Run `python check_data.py` and confirm all four O'Neil entries are `[OK]`.
3. Create a new empty GitHub repository named `SAFIR`.
4. Upload the **contents** of this folder (not the outer zip file).
5. Replace `USERNAME` in `CITATION.cff` with the GitHub account or organization name.
6. After the repository is public, place its URL in the manuscript Data and Software Availability section.

## Environment files

- `requirements.txt` contains the minimal pinned dependencies needed by the released modular O'Neil workflow.
- `requirements_full.txt` records the full Python package environment supplied from the original `safir` Jupyter environment. A machine-specific editable `minervachem` path was removed because it is not portable and is not imported by the released modular O'Neil workflow.
- `environment.yml` records Python 3.10.20 and the minimal pinned dependencies.

The large DrugComb tables are gzip-compressed so individual repository files remain suitable for GitHub upload.
