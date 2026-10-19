# Data and evaluation protocol

## Decision and cohort

The row grain is one `msno` in `train_v2.csv`, with `is_churn` as the label. The illustrative decision cutoff is 2017-02-28 for the subsequent March label cohort. Every event feature uses transaction or listening dates on or before that cutoff. The label is never part of model input. The code rejects duplicate or invalid labels and event files with no matching pre-cutoff behavior.

The data lacks an ingestion timestamp. The filter checks **event time**, not whether each event was already available to a live marketing system on the cutoff date. Membership attributes from `members_v3.csv` are excluded because the file does not establish their historical state at that cutoff.

## Features

| Group | Features | Rule |
| --- | --- | --- |
| Transactions | count, sum of amount paid, days since latest transaction, latest plan days, latest auto-renew flag | Include only `transaction_date <= snapshot`. |
| Listening | count of log rows, total listening seconds, total unique tracks, days since latest log | Include only `date <= snapshot`. |

`log_days` currently counts valid log rows, which should be interpreted as active days only if the source has one row per user-day. Missing history is represented by zero counts and `-1` recency; future work should distinguish true inactivity from incomplete ingestion. No demographic, member status, expiration-date, cancellation, or post-cutoff field is used.

## Model and split

The estimator is a scikit-learn pipeline: median imputation, standard scaling, and logistic regression. Preprocessing is fitted on the 75% training partition only. The remaining 25% is a stratified random holdout, selected with seed 42. No hyperparameter or threshold selection is made on that holdout. A comparator predicts the training partition's churn prevalence for every held-out user.

This design answers a limited question: can pre-cutoff features separate labels in a held-out part of the **same March cohort**? A production decision would need a separate out-of-time cohort, stability checks, calibration review, and an intervention study. Repeated users across time, delayed data arrival, and the competition's exact label-generation rule need explicit review before stronger claims.

## Result acceptance

Before filling [RESULTS.md](RESULTS.md), verify the official files and their dates, record the source file sizes and checksums locally, and run the command on the complete chosen cohort. Save the exact code commit, Python/package versions, snapshot, counts, class prevalence, holdout metrics, and any exclusions. Inspect whether the event sources genuinely contain pre-cutoff records for the labeled population. Report a confidence interval or a second untouched cohort before making comparative claims. Keep raw data, IDs, predictions, and model artifacts out of the public repository.

## References

- [KKBox competition data](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data) and [rules](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules)
- [Kaggle discussion of the file versions](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/discussion/44363)
- [scikit-learn: data leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
