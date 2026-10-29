# Verification status 

**Last full check: 2026-10-29, Windows, Anaconda base environment.**

## Executed successfully

- `python verify_project.py` (full run, not `--data-only`) printed both PASS lines and "Verification complete":
  - Static API, dependency, licensing and model-data checks (26 Plotly calls, no deprecated width calls, data and report consistency).
  - Streamlit `AppTest` on the default and Plotly globe routes, with no warning, error or exception elements.
- Package versions at the time of the run: pandas 2.3.3, numpy 2.3.5, scikit-learn 1.8.0, Streamlit 1.66.1, Plotly 6.5.2. The checker fails unless Streamlit is 1.66.1.
- The dashboard starts with `python -m streamlit run app.py` and serves on port 8501.
- The same checker also passed in a separate run on Linux (Python 3.12) with pandas 2.2.3 and with pandas 2.3.3.

## Dependency note

`requirements.txt` pins `pandas==2.3.3`, because pandas 2.2.3 has no prebuilt Windows package for Python 3.14. `verify_project.py` checks for this pin.

## Earlier build-environment checks

These were run before Streamlit was available in the build environment:

- Python compilation of the application and project scripts.
- `python verify_project.py --data-only`.
- The revised `01_data_cleaning.ipynb` code cells, using only the bundled OpenFootball JSON files.
- `python train_model.py` and `python audit_model.py` from the cleaned OpenFootball data, reproducing the 2026 accuracy (61/104, 58.7%) and log loss (0.928).

