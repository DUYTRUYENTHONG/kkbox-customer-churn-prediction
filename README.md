# KKBox customer churn: snapshot-safe baseline

An independent Python baseline for the [WSDM KKBox churn challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge). It turns pre-snapshot transactions and listening history into customer features, trains logistic regression, and reports holdout probability and ranking metrics. The repository is prepared for Ali Nguyen with Codex assistance.

**Project status:** working source and synthetic-fixture tests. An evaluation on the official KKBox files has **not yet been run** in this repository. There is no competition submission, verified KKBox score, production model, or measured retention impact. See [data and evaluation design](docs/DATA_AND_EVALUATION.md) and [results status](docs/RESULTS.md).

## Why this baseline exists

A churn decision made at the end of February 2017 cannot use March 2017 events. This pipeline accepts a snapshot date and filters every transaction and listening event to `event_date <= snapshot`. A synthetic test adds events after that date and verifies they cannot enter the features. The local model then uses a fixed, stratified 25% holdout from the March label cohort. That split measures discrimination within the cohort; it is **not** an out-of-time or causal evaluation.

The first draft paired `train_v2.csv` with `transactions_v2.csv` and `user_logs_v2.csv` at a `2017-02-28` cutoff. Those `_v2` event files are March updates, making that pairing unsuitable for pre-cutoff behavior. This version uses the earlier `transactions.csv` and `user_logs.csv` files and rejects an input whose labeled users have no matching pre-snapshot transactions or logs. The [Kaggle discussion on file versions](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/discussion/44363) is useful context; the exact input dates must still be checked during a real run.

## Obtain the data

Download the official files from [Kaggle's competition page](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data) after accepting its rules. The [competition rules](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules) restrict redistribution. Keep extracted CSVs outside Git or under ignored `data/raw/`:

| File | Role |
| --- | --- |
| `train_v2.csv` | March churn labels: `msno`, `is_churn` |
| `transactions.csv` | Earlier transaction history: `msno`, `transaction_date`, `actual_amount_paid`, `payment_plan_days`, `is_auto_renew` |
| `user_logs.csv` | Earlier listening history: `msno`, `date`, `total_secs`, `num_unq` |

`members_v3.csv` is intentionally omitted from this baseline: its member attributes are not versioned as of the snapshot, so their availability at the decision date is not demonstrated. Raw customer IDs, rows, trained models, and derived customer-level features must not be committed.

## Reproduce

Requires Python 3.11+ and enough disk space for the extracted competition files. The event CSVs are streamed, but customer aggregates for labeled users are held in memory. Full-data runtime and memory usage remain unmeasured.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
kkbox-churn --labels data/raw/train_v2.csv \
  --transactions data/raw/transactions.csv \
  --logs data/raw/user_logs.csv \
  --snapshot 2017-02-28 --output outputs
```

On PowerShell, use `.venv\Scripts\python.exe -m pip install -e ".[test]"`, `.venv\Scripts\python.exe -m pytest`, and `.venv\Scripts\kkbox-churn.exe` for the final command. Outputs are ignored by Git: `outputs/metrics.json` and `outputs/baseline.joblib`. A real-data result should be published only after recording the run date, source file versions, row counts, machine and software versions, code commit, exact split, metrics, and limitations in [docs/RESULTS.md](docs/RESULTS.md). Do not publish customer-level predictions.

## Metrics and interpretation

The CLI reports holdout log loss (the challenge's probability metric), ROC AUC, average precision, Brier score, churn prevalence, and the observed churn rate among the highest-scored 10% of held-out users. It also reports pre-snapshot event counts, users without event history, and the log loss of a training-prevalence-only comparator. Thresholds and outreach actions are not tuned in this baseline. Better ranking on a holdout would not demonstrate that contacting a user prevents churn.

## Source and credit

This implementation is original code informed by the MIT-licensed [veera491/customer-churn-prediction-kkbox](https://github.com/veera491/customer-churn-prediction-kkbox) academic repository and by the competition data description. See [NOTICE.md](NOTICE.md). None of that project's code, report, notebook, or reported model results is presented as Ali Nguyen's work. The project's MIT license applies to this source code, **not** to the KKBox competition data.
