# Data and evaluation protocol

## Decision and cohort

The row grain is one `msno` in `train_v2.csv`, with `is_churn` as the label. The competition describes this as its March 2017 churn-label cohort. The decision cutoff for this baseline is 2017-02-28, using the original `transactions.csv` and `user_logs.csv` histories that end on that date. Every event feature uses transaction or listening dates on or before the cutoff. The label is never part of model input. The code rejects duplicate or invalid labels, missing required CSV columns, malformed rows, and event files with no matching pre-cutoff behavior. It does not independently regenerate the competition labels.

The data lacks an ingestion timestamp. The filter checks **event time**, not whether each event was already available to a live marketing system on the cutoff date. Membership attributes from `members_v3.csv` are excluded because the file does not establish their historical state at that cutoff.

## Features

| Group | Features | Rule |
| --- | --- | --- |
| Transactions | count, sum of amount paid, days since latest transaction, latest plan days, latest auto-renew flag | Include only `transaction_date <= snapshot`. |
| Listening | count of log rows, total listening seconds, total unique tracks, days since latest log | Include only `date <= snapshot`. |

For transactions sharing the latest date, the measured DuckDB engine chooses the row with the greatest plan length, then auto-renew flag. This explicit tie rule makes the two latest-transaction fields come from one row and keeps reruns deterministic. The Python streaming engine follows the last such row in file order, so exact score reproduction should use `--engine duckdb`.

The feature is named `log_rows` because the source's one-row-per-user-day behavior has not been verified here. Missing history is represented by zero counts and `-1` recency; future work should distinguish true inactivity from incomplete ingestion. No demographic, member status, expiration-date, cancellation, or post-cutoff field is used.

The normal training command audits coverage before model fitting and includes aggregate diagnostics in `metrics.json`: rows read, rows matching labeled users, invalid-date and post-snapshot exclusions, usable pre-snapshot rows, and labeled users without each event type. `--audit-only` writes `audit.json` without training, even when a source has zero usable pre-snapshot rows; it requires another full scan if training is run afterward. `--engine duckdb` performs the full-data scan with a disk-backed native engine, while `--engine python` keeps the original streaming implementation. The counts help detect wrong file versions and missing coverage without publishing customer IDs. They do not prove that every event was available to an operational system at the cutoff.

## Model and split

The estimator is a scikit-learn pipeline: median imputation, standard scaling, and logistic regression. Preprocessing is fitted on the 75% training partition only. The remaining 25% is a stratified random holdout, selected with seed 42. No hyperparameter or threshold selection is made on that holdout. A comparator predicts the training partition's churn prevalence for every held-out user.

This design answers a limited question: can pre-cutoff features separate labels in a held-out part of the **same labeled cohort**? A production decision would need a separate out-of-time cohort, stability checks, calibration review, and an intervention study. Repeated users across time, delayed data arrival, and the competition's exact label-generation rule need explicit review before stronger claims.

## Result acceptance

The verified run, input hashes, cohort and source counts, software versions, exact code commit, and held-out metrics are recorded in [RESULTS.md](RESULTS.md). The official event files contributed pre-cutoff records for the labeled population. A confidence interval or an untouched later cohort is still needed before comparative or production claims. Keep raw data, IDs, predictions, DuckDB working files, and model artifacts out of the public repository.

## References

- [KKBox competition data](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data) and [rules](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules)
- [Kaggle discussion of the file versions](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/discussion/44363)
- [Competition host clarification on churn labels and updated data](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/discussion/43622)
- [scikit-learn: data leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
