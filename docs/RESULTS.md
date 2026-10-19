# KKBox evaluation results

## Verified local run

This is a **local, within-cohort holdout**, using the official [KKBox competition files](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data). It is not a Kaggle leaderboard submission, an out-of-time validation, a production model, or evidence that an intervention reduces churn. No customer-level data, predictions, or model artifact is published here.

| Run detail | Value |
| --- | --- |
| Run date | 2026-10-19 (Asia/Bangkok) |
| Model/source commit | `052d4b5a977e118025a81523dd7c608631d2a673` |
| Environment | Windows; Python 3.12.14; pandas 2.3.3; scikit-learn 1.9.1; DuckDB 1.5.6; joblib 1.6.0 |
| Machine | 15.5 GiB RAM, 18 logical processors; DuckDB limited to 8 GB and four threads |
| Snapshot | 2017-02-28; only events dated on or before this date enter features |
| Labels | `train_v2.csv`, the competition's March 2017 churn-label cohort |
| Split | One stratified random 75%/25% train/holdout split, seed 42; preprocessing fitted on train only |
| Model | Median imputation, standard scaling, logistic regression (`max_iter=1000`), no tuning on holdout |
| Full-run wall time | 326.43 seconds, including both CSV feature scans, model fitting, and evaluation |
| Memory | Exact peak unmeasured; highest sampled process working set was approximately 3.7 GB |

### Input provenance

The files were downloaded after the account gained competition access. These are hashes of the **extracted CSVs**, so another authorized user can verify the exact inputs locally. The raw files remain under ignored `data/raw/` and are not redistributed.

| Official file | Bytes | SHA-256 |
| --- | ---: | --- |
| `train_v2.csv` | 45,635,134 | `1f7590a3467174f5a147db3fb85dd1393074b6c13da0fd6e70af31bf37dee5cc` |
| `transactions.csv` | 1,729,298,376 | `537c269140742fc74b28ba83941dff7b0ca00974dd48b80994948b8917adddef` |
| `user_logs.csv` | 30,514,081,415 | `32d83038c86f7974b8f346bb7da1876a186248aa42c74a432517e70cf514d729` |

### Audit and scores

| Audit measure | Full three-source run |
| --- | ---: |
| Labeled users / churn labels | 970,960 / 87,330 |
| Transaction rows read / matched labeled users | 21,547,746 / 15,123,586 |
| Listening rows read / matched labeled users | 392,106,543 / 236,970,464 |
| Usable pre-snapshot transaction / listening rows | 15,123,586 / 236,970,464 |
| Matched rows after cutoff or with invalid dates | 0 in each source |
| Labeled users with no transaction / listening history | 2,527 / 120,664 |

| Held-out measure | Transactions + listening | Transactions only | Train-prevalence comparator |
| --- | ---: | ---: | ---: |
| Users (train / holdout) | 728,220 / 242,740 | 728,220 / 242,740 | Same split |
| Train / holdout churn prevalence | 0.089943 / 0.089940 | 0.089943 / 0.089940 | Same split |
| Log loss ↓ | **0.217324** | 0.218071 | 0.302399 |
| ROC AUC ↑ | 0.819295 | 0.814133 | Not used |
| Average precision ↑ | 0.523279 | 0.518909 | Not used |
| Brier score ↓ | 0.057714 | 0.057876 | Not used |
| Churn rate in highest-scored 10% | 0.483315 | 0.479937 | Not used |

The highest-scored decile contains 24,274 held-out users. The transaction-only run is a deliberately narrower baseline from the same labels and split; it does not use listening features. These descriptive comparisons do not include confidence intervals or prove improvement on a future cohort.

### Reproduce with authorized data

```sh
kkbox-churn --labels data/raw/train_v2.csv \
  --transactions data/raw/transactions.csv \
  --logs data/raw/user_logs.csv \
  --snapshot 2017-02-28 --engine duckdb --output outputs/full-data-final
```

The command writes aggregate `metrics.json`, a local `baseline.joblib`, and a local DuckDB feature cache beneath the ignored output folder. Its feature scan checks cutoff coverage before fitting the model. To reproduce the narrower baseline, replace `--logs ...` with `--transactions-only` and use a separate output directory. The exact metrics JSON files are local outputs; the tables above transcribe their aggregate values.

### Interpretation limits

- The holdout samples the same March-label cohort at random. An independent later cohort, ideally April labels with an appropriate March cutoff, is needed for an out-of-time estimate.
- The code filters by event date. The files do not establish when each event arrived in an operational marketing system, so real-time availability at the cutoff is unverified.
- The supplied competition labels are used as given; this repository does not reimplement the host's per-user label generator. The pipeline excludes `members_v3.csv`, cancellation, and expiration-date fields to avoid unproven historical state or label leakage.
- About 12.4% of labeled users have no matching listening history. Zero counts and `-1` recency encode that absence; they do not distinguish inactivity from missing collection.
- No threshold, outreach policy, confidence interval, repeated split, calibration study, or intervention test was performed. A higher ranking metric alone does not establish business value.
