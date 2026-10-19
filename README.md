# KKBox customer churn: snapshot-safe baseline

An independent Python baseline for the [WSDM KKBox churn challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge). It turns pre-snapshot transactions and listening history into customer features, trains logistic regression, and reports holdout probability and ranking metrics. The repository is prepared for Ali Nguyen with Codex assistance.

**Project status:** a local run on the official KKBox files is complete. The three-source baseline reached **0.217324 holdout log loss** and **0.819295 ROC AUC** on a fixed, stratified 25% split of 970,960 labeled users. These are within-cohort results, not a Kaggle leaderboard score, out-of-time validation, production model, or measured retention impact. See the [full audit, input hashes, and limitations](docs/RESULTS.md) and [evaluation design](docs/DATA_AND_EVALUATION.md).

## Why this baseline exists

A churn decision made at the end of February 2017 cannot use March 2017 events. This pipeline accepts a snapshot date and filters every transaction and listening event to `event_date <= snapshot`. A synthetic test adds events after that date and verifies they cannot enter the features. The local model then uses a fixed, stratified 25% holdout from the `train_v2.csv` March-label cohort. That split measures discrimination within the cohort; it is **not** an out-of-time or causal evaluation. The competition's supplied labels are used as given; this project does not reimplement their label generator.

The first draft paired `train_v2.csv` with `transactions_v2.csv` and `user_logs_v2.csv` at a `2017-02-28` cutoff. Those `_v2` event files are March updates, making that pairing unsuitable for pre-cutoff behavior. The verified run uses the earlier `transactions.csv` and `user_logs.csv` files. Its audit found 15,123,586 matching pre-snapshot transaction rows, 236,970,464 matching pre-snapshot listening rows, and zero matching rows after the cutoff. See the [official file descriptions](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data) and [Kaggle discussion on file versions](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/discussion/44363).

## Obtain the data

Download the official files from [Kaggle's competition page](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data) after accepting its rules. The [competition rules](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules) restrict redistribution. Keep extracted CSVs outside Git or under ignored `data/raw/`:

| File | Role |
| --- | --- |
| `train_v2.csv` | March churn labels: `msno`, `is_churn` |
| `transactions.csv` | Earlier transaction history: `msno`, `transaction_date`, `actual_amount_paid`, `payment_plan_days`, `is_auto_renew` |
| `user_logs.csv` | Earlier listening history: `msno`, `date`, `total_secs`, `num_unq` |

`members_v3.csv` is intentionally omitted from this baseline: its member attributes are not versioned as of the snapshot, so their availability at the decision date is not demonstrated. Raw customer IDs, rows, trained models, and derived customer-level features must not be committed.

## Reproduce

Requires Python 3.11+ and enough disk space for the extracted competition files. The DuckDB engine scans the large CSVs in parallel and keeps its local working database under the ignored output directory. A Python streaming engine remains available as `--engine python`; it is slower on the full dataset. The verified DuckDB run took 326.43 seconds on a 15.5 GiB RAM Windows machine.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
kkbox-churn --labels data/raw/train_v2.csv \
  --transactions data/raw/transactions.csv \
  --logs data/raw/user_logs.csv \
  --snapshot 2017-02-28 --engine duckdb --output outputs/full-data
```

The command audits source coverage and the date cutoff before fitting, then writes aggregate `outputs/full-data/metrics.json`, a local `baseline.joblib`, and a local DuckDB feature cache. All are ignored by Git. To inspect counts without fitting, add `--audit-only`; that separate run scans the CSVs again. For a smaller transaction-only baseline, replace `--logs ...` with `--transactions-only` and choose a different output directory. On PowerShell, use `.venv\Scripts\python.exe -m pip install -e ".[test]"`, `.venv\Scripts\python.exe -m pytest`, and `.venv\Scripts\kkbox-churn.exe` for the final command. Do not publish customer-level predictions or trained artifacts.

## Metrics and interpretation

The CLI reports holdout log loss (the challenge's probability metric), ROC AUC, average precision, Brier score, churn prevalence, and the observed churn rate among the highest-scored 10% of held-out users. It also reports CSV row counts, labeled-user matches, post-snapshot and invalid-date exclusions, pre-snapshot event counts, users without event history, and the log loss of a training-prevalence-only comparator. Missing required columns and malformed CSV rows stop the run. These aggregate diagnostics contain no customer IDs. Thresholds and outreach actions are not tuned in this baseline. Better ranking on a holdout would not demonstrate that contacting a user prevents churn.

## Source and credit

This implementation is original code informed by the MIT-licensed [veera491/customer-churn-prediction-kkbox](https://github.com/veera491/customer-churn-prediction-kkbox) academic repository and by the competition data description. See [NOTICE.md](NOTICE.md). None of that project's code, report, notebook, or reported model results is presented as Ali Nguyen's work. The project's MIT license applies to this source code, **not** to the KKBox competition data.
