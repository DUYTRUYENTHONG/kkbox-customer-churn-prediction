# KKBox customer churn prediction baseline

An independent, local machine-learning baseline by **Ali Nguyen** for the
[KKBox churn challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge).
It builds customer features from member, transaction, and listening-log CSVs using
only events on or before a chosen snapshot date, then evaluates a logistic-regression
model on a stratified holdout. The challenge uses log loss as its evaluation metric.

**Status:** working baseline code, not a production system or a verified competition
result. No real KKBox data or model artifacts are included. The CV's LightGBM,
Gemini, AUC, recall, and revenue-at-risk claims are not results of this repository.

## Run locally

Use Python 3.11 or newer. Download the dataset directly from Kaggle after accepting
its terms; do not commit or redistribute the competition data.

```sh
python -m venv .venv
python -m pip install -e '.[test]'
pytest
kkbox-churn --labels data/raw/train_v2.csv --members data/raw/members_v3.csv \
  --transactions data/raw/transactions_v2.csv --logs data/raw/user_logs_v2.csv \
  --snapshot 2017-02-28 --output outputs
```

On Windows PowerShell, run `.venv\Scripts\python.exe -m pip install -e '.[test]'`
and `.venv\Scripts\kkbox-churn.exe` instead of the shell commands above.
The command writes `outputs/metrics.json` and `outputs/baseline.joblib` locally.

## Data contract and limitations

- `labels`: `msno`, `is_churn`; each user should appear once.
- `members`: `msno`, `city`, `registered_via`.
- `transactions`: `msno`, `transaction_date`, `actual_amount_paid`,
  `payment_plan_days`, `is_auto_renew`.
- `logs`: `msno`, `date`, `total_secs`, `num_unq`.
- Date columns use `YYYYMMDD`. The caller must choose a snapshot that precedes
  the outcomes represented by the labels. Rows after the snapshot are excluded.
- This baseline streams the two event CSVs but holds customer aggregates in memory.
  Real-dataset runtime and scores have not yet been measured.
- The random holdout is a local baseline, not a competition submission or a
  temporal production validation. It does not establish business impact.

## Attribution

The project was informed by the MIT-licensed
[veera491 academic repository](https://github.com/veera491/customer-churn-prediction-kkbox).
See [NOTICE.md](NOTICE.md). Its code, notebook, thesis report, and reported results
are not presented here as Ali Nguyen's work.
