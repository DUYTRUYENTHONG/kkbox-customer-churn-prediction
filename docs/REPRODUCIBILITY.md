# Reproduce and inspect the baseline

Run these commands from the repository root. Download the three extracted CSVs yourself from the [official KKBox competition data page](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data) after accepting its rules. The files are intentionally excluded from this repository. Do not commit raw data, customer IDs, customer-level features or predictions, `baseline.joblib`, or the DuckDB cache.

## 1. Check inputs and environment

Use Python 3.11 or newer. The three extracted CSVs total more than 32 GB. Allow additional space for the virtual environment, DuckDB working database, and model output. The published run used Python 3.12.14, DuckDB 1.5.6, pandas 2.3.3, scikit-learn 1.9.1, and joblib 1.6.0 on Windows. `pyproject.toml` supplies supported dependency ranges rather than an exact lockfile; different resolved versions can change floating-point results.

Place `train_v2.csv`, `transactions.csv`, and `user_logs.csv` under `data/raw/`, or pass absolute paths to the CLI. Check the extracted files against [the recorded byte sizes and SHA-256 values](RESULTS.md#input-provenance). On Windows PowerShell:

```powershell
Get-Item data\raw\train_v2.csv, data\raw\transactions.csv, data\raw\user_logs.csv | Select-Object Name, Length
Get-FileHash data\raw\train_v2.csv, data\raw\transactions.csv, data\raw\user_logs.csv -Algorithm SHA256 | Select-Object Path, Hash
```

On macOS/Linux:

```sh
sha256sum data/raw/train_v2.csv data/raw/transactions.csv data/raw/user_logs.csv
```

If a hash differs, confirm that the files are the extracted official CSVs and have not been edited. Do not treat a differing run as an exact reproduction.

## 2. Install and test

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
.venv\Scripts\python.exe -m pytest
```

macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

The tests use tiny synthetic CSVs, so they do not require Kaggle access. They check snapshot exclusion, audit counts, bad input rejection, deterministic DuckDB transaction ties, CLI outputs, and model metric creation. They do **not** constitute an independent full-data result. Continuous integration runs the same test suite on the public source.

## 3. Run the published configuration

PowerShell:

```powershell
.venv\Scripts\kkbox-churn.exe --labels data\raw\train_v2.csv --transactions data\raw\transactions.csv --logs data\raw\user_logs.csv --snapshot 2017-02-28 --engine duckdb --output outputs\full-data-final
```

macOS/Linux:

```sh
.venv/bin/kkbox-churn --labels data/raw/train_v2.csv --transactions data/raw/transactions.csv --logs data/raw/user_logs.csv --snapshot 2017-02-28 --engine duckdb --output outputs/full-data-final
```

The normal run scans inputs, checks pre-snapshot source coverage, fits on the training partition, scores the holdout, and prints the same aggregate JSON it saves as `metrics.json`. The 2026-10-19 reference result was generated from model/source commit `052d4b5a977e118025a81523dd7c608631d2a673`; later documentation commits do not change that code. See [RESULTS.md](RESULTS.md) for the expected audit counts, scores, versions, and limitations. The run took 326.43 seconds on the reported Windows machine; that is an observation, not a performance guarantee.

## CLI options

| Option | Required? | Meaning |
| --- | --- | --- |
| `--labels PATH` | Yes | CSV containing `msno,is_churn` |
| `--transactions PATH` | Yes | Pre-cutoff `transactions.csv` |
| `--logs PATH` | One of this or `--transactions-only` | Pre-cutoff `user_logs.csv` |
| `--transactions-only` | One of this or `--logs` | Omit all listening features; useful for the narrower comparison |
| `--snapshot YYYY-MM-DD` | Yes | Inclusive event-date cutoff |
| `--engine python\|duckdb` | No | Defaults to `python`; choose `duckdb` for the published full-data run |
| `--output DIR` | No | Defaults to `outputs/`; choose a distinct directory per run |
| `--audit-only` | No | Write coverage diagnostics without training; a later training command rescans the files |

For a transaction-only comparison, replace `--logs PATH` with `--transactions-only` and use a separate `--output` directory. For a coverage-only check, add `--audit-only` to the full-data command. The latter writes `audit.json` and no model or holdout metrics. It also allows zero pre-snapshot coverage so wrong file versions can be identified.

## Output contract

| Local file | Produced by | Contents |
| --- | --- | --- |
| `metrics.json` | Training run | Aggregate cohort, audit, and holdout measures; selected engine, source basenames, and snapshot |
| `baseline.joblib` | Training run | Fitted scikit-learn pipeline; private model artifact, not safe to load from untrusted sources |
| `audit.json` | `--audit-only` | Aggregate source counts and run configuration; no model metrics |
| `feature_cache.duckdb` | DuckDB engine | Working tables with matched user IDs and derived values; keep private |

The output folder is ignored by Git. The JSON audit names include `*_rows_read` (all rows scanned), `*_rows_for_labeled_users` (matched rows), `*_rows_invalid_date` and `*_rows_after_snapshot` (matched exclusions), and `pre_snapshot_*_rows` (matched, valid, eligible rows). `users_without_*` counts labeled users with no eligible rows from that source. Do not infer that unmatched rows were invalid or post-snapshot; they were not examined for those properties.

The metric keys are `holdout_log_loss` (lower is better), `holdout_auc_roc` and `holdout_average_precision` (higher is better), `holdout_brier_score` (lower is better), and `top_decile_churn_rate` for the observed churn fraction among the highest-scored 10% of held-out users. `prevalence_log_loss` comes from predicting the training-set churn rate for every holdout user. The [evaluation protocol](DATA_AND_EVALUATION.md) defines the split and the limits of this comparison.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No matching pre-snapshot transactions or logs | Use `transactions.csv` and `user_logs.csv`, not the March `_v2` updates; run `--audit-only` to inspect aggregate coverage. |
| Missing required CSV columns or invalid labels | Confirm the extracted official file and its hash. Do not silently rename or repair fields. |
| Invalid pre-snapshot numeric rows | The DuckDB run stops rather than fitting on silently coerced values. Verify file integrity and report the anomaly. |
| Disk or memory pressure | Check free disk space for the >32 GB extracted inputs and DuckDB cache; the DuckDB implementation is configured for four threads and an 8 GB limit. |
| Scores differ from the published table | Check input hashes, exact source commit, `--engine duckdb`, cutoff, feature mode, Python/library versions, and that the output came from a complete run. |

## What a reproduction does not prove

Matching the recorded holdout numbers confirms this implementation and input snapshot, within version-dependent numeric variation. It does not provide a Kaggle leaderboard result, an untouched future cohort, an intervention experiment, or a deployable retention policy. Those require separate data access, evaluation design, and reporting.
