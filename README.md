# Direct Blowcount Prediction — prototype app

Companion tool to the manuscript. It loads the trained model and reproduces
the paper's feature pipeline, so predictions match the reported results.

## Setup

1. Put these three files in one folder:
   - `app.py`
   - `blowcount_model.joblib`   (saved from the analysis notebook)
   - `requirements.txt`

2. Install dependencies:
   pip install -r requirements.txt

3. Run:
   streamlit run blowcount_app.py

A browser tab opens at http://localhost:8501

## Using it

- **Load worked example** — runs immediately on a synthetic sand profile with
  a clay interval (mimics the blind test site). No file needed.
- **Upload CPT file** — whitespace- or tab-delimited, either:
  - 3 columns: Depth, qt, fs, or
  - 5 columns: Depth, qc, qc, fs, fs   (as in the training data)
- Set the hammer, the simulated SRD, and the target embedment in the sidebar.

## Honest reporting

The prediction band is nominally 80% but achieved **61% empirical coverage**
on the blind holdout site. The app displays the empirical figure. On geologies
unlike the two training sites the band is indicative only, not calibrated.
This tool is a research prototype, not validated for design use.
