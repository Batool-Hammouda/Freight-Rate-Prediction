# Freight Rate Prediction Challenge

This project predicts shipment prices using LightGBM on rate per mile, then multiplies by distance. Cleaning and features live only in `src/features.py`; notebook and command-line training import that module. [Source: src/features.py; src/train.py]

## Data and findings

`train_test.csv`: 48,000 loads [Source: reports/evidence/training_summary.json]. `validation.csv`: 12,000 loads [Source: reports/evidence/external_summary.json]. The source CSVs retain their contents under normalized underscore filenames; `score.py` is unchanged. [Source: reports/evidence/input_hashes.json]

Weight signs are corrected under a documented sign-error assumption; missing weight and market index remain NaN with flags. No rows are dropped or imputed. Quote behaviour changes by month, so quote-derived features are excluded. Coordinates represent locations, including unseen city names. [Source: src/features.py; reports/evidence/weight_comparison.csv; reports/evidence/training_behaviour.csv; reports/evidence/external_summary.json]

## Setup and rebuild

The tested Python environment and package versions are recorded in `reports/evidence/environment.json`; dependencies are pinned in `requirements.txt`.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux, then:

```bash
python -m pip install -r requirements.txt
python -m src.run
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`src.run` refits with the locked settings, saves the model and feature list, merges predictions into template order, fills the December file from its preserved backup, and invokes the scorer. It does not repeat model selection or the held-out evaluation. [Source: src/run.py]

The notebook can be opened in Jupyter or executed with:

```bash
python tools/execute_notebook.py
```

The notebook displays the saved development and held-out evidence, recomputes diagnostics, and rebuilds final predictions. The fresh-kernel execution does not rescore the held-out period. The evaluation implementation is `src.evaluate`; its ledger intentionally rejects repeated selection after final test exposure. [Source: tools/execute_notebook.py; src/evaluate.py]

## Notebook

`notebooks/assessment.ipynb` is the final executed analysis. `tools/execute_notebook.py` executes it from a fresh kernel. To open it interactively:

```bash
python -m jupyterlab notebooks/assessment.ipynb
```

The final report is `reports/report.pdf`.

## Folder structure

```text
README.md
requirements.txt
.gitignore
score.py
data/
notebooks/assessment.ipynb
src/
tools/execute_notebook.py
models/
reports/report.pdf
reports/evidence/
scorer_results/candidate_december.png
validation_predictions.csv
```

`reports/evidence/` contains the audit trail and result tables. `data/december_chart_inputs.original.csv` preserves the supplied empty input. [Source: src/run.py]

## Validation and results

January-August development and September-October test are separated by date. Expanding folds evaluate April-August using earlier months only. August is closest to the external quote pattern. The current LightGBM test was evaluated once after choices were locked. Earlier Ridge work and exploratory analysis had already exposed the test months, so this is not a historically untouched holdout. [Source: reports/evidence/split.csv; reports/evidence/cv_folds.csv; reports/evidence/training_behaviour.csv; reports/evidence/external_behaviour.csv; reports/evidence/holdout_ledger.json]

| Evaluation | MAE $ | MAPE | RMSE $ | R2 |
|---|---|---|---|---|
| Default final folds | 108.12 [Source: reports/evidence/cv_summary.csv] | 4.71% [Source: reports/evidence/cv_summary.csv] | 607.94 [Source: reports/evidence/cv_summary.csv] | 0.84 [Source: reports/evidence/cv_summary.csv] |
| Selected folds | 104.99 [Source: reports/evidence/selected_cv_folds.csv] | 4.57% [Source: reports/evidence/selected_cv_folds.csv] | 607.22 [Source: reports/evidence/selected_cv_folds.csv] | 0.84 [Source: reports/evidence/selected_cv_folds.csv] |
| Selected August | 109.26 [Source: reports/evidence/selected_cv_folds.csv] | 4.93% [Source: reports/evidence/selected_cv_folds.csv] | 620.67 [Source: reports/evidence/selected_cv_folds.csv] | 0.82 [Source: reports/evidence/selected_cv_folds.csv] |
| Selected test | 160.21 [Source: reports/evidence/holdout_metrics.csv] | 6.78% [Source: reports/evidence/holdout_metrics.csv] | 648.41 [Source: reports/evidence/holdout_metrics.csv] | 0.82 [Source: reports/evidence/holdout_metrics.csv] |


The bounded tuning check selected alpha 0.5 [Source: models/selection.json]. Extra features failed the acceptance rule. All trials and both seeds are recorded in `reports/evidence/tuning.csv` and `tuning_folds.csv`. Scores use MAE for selection; the assessors' metric is not exposed. [Source: src/evaluate.py; score.py]

## Expected outputs

- `validation_predictions.csv`: 12,000 positive predictions in template ID order [Source: reports/evidence/prediction_summary.json].
- `data/december_chart_inputs.csv`: 31 filled predictions with original fixed columns preserved [Source: reports/evidence/prediction_summary.json].
- `models/final_model.joblib`, `models/features.json`, and `models/final_settings.json`.
- `scorer_results/candidate_december.png`: scorer-produced chart, ranging from $802.49 [Source: reports/evidence/prediction_summary.json] to $818.45 [Source: reports/evidence/prediction_summary.json].
- `reports/report.pdf`: final report with validation results and the December chart.

The scorer checks file validity and produces a chart; it does not report predictive accuracy. [Source: score.py]

## Limitations

The test-period error exceeds the development-validation average; its cause was not investigated, and the November-December error could also be higher. [Source: reports/evidence/holdout_metrics.csv; reports/evidence/selected_cv_folds.csv] The sign correction and market-index availability at quote time are assumptions. August is the only flat-behaviour month in train_test.csv. Prices for validation.csv cannot be verified. The unseen-city check is one random draw. December changes only through daily market index calculated from supplied same-day external loads. [Source: reports/evidence/weight_comparison.csv; reports/evidence/training_behaviour.csv; reports/evidence/external_summary.json; reports/evidence/unseen_city_test.csv; src/run.py]

## Submission materials

Push this folder as the repository root. The submission deliverables are the repository, `validation_predictions.csv`, `reports/report.pdf`, and your separately recorded Loom video link.

## Saved evaluation evidence

`reports/evidence/` is part of the project. It contains the saved model comparisons and held-out results read by the notebook, the data-quality findings, and the sources cited by the PDF. Keeping these files lets reviewers inspect the recorded results without repeating model selection or the held-out evaluation.

`final_checks.txt` records prediction-file checks. `fresh_copy_check.json` records the successful clean-environment rebuild before removal of the report-authoring dependency. `notebook_execution.json` records the completed fresh-kernel run. `environment.json` records that run's environment; the current `requirements.txt` omits the now-unused Word report dependency.
