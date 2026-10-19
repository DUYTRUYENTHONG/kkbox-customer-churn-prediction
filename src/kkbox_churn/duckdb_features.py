"""Parallel, disk-backed feature extraction for the full KKBox CSV files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


def build_features_with_audit_duckdb(
    labels_path: Path,
    transactions_path: Path,
    logs_path: Path | None,
    snapshot: datetime,
    work_dir: Path,
    *,
    require_coverage: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Scan each event CSV once; keep the working database outside version control."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cutoff = snapshot.date().isoformat()
    with duckdb.connect(str(work_dir / "feature_cache.duckdb")) as db:
        db.execute("SET memory_limit = '8GB'")
        db.execute("SET threads = 4")
        db.execute(
            """CREATE OR REPLACE TABLE labels AS
               SELECT msno, CAST(is_churn AS INTEGER) AS is_churn
               FROM read_csv(?, header=true, all_varchar=true)""",
            [str(labels_path)],
        )
        label_rows, churn_labels, unique_users, bad_labels = db.execute(
            """SELECT count(*), sum(is_churn), count(DISTINCT msno),
                      count(*) FILTER (WHERE msno IS NULL OR msno = '' OR is_churn NOT IN (0, 1))
               FROM labels"""
        ).fetchone()
        if not label_rows or unique_users != label_rows or bad_labels:
            raise ValueError("Labels must have one nonempty row per msno and binary targets")

        print("Scanning transactions with DuckDB", flush=True)
        db.execute(
            """CREATE OR REPLACE TABLE tx_agg AS
               WITH parsed AS (
                   SELECT l.msno AS user_id,
                          TRY_STRPTIME(t.transaction_date, '%Y%m%d')::DATE AS event_date,
                          t.actual_amount_paid, t.payment_plan_days, t.is_auto_renew
                   FROM read_csv(?, header=true, all_varchar=true) AS t
                   LEFT JOIN labels AS l ON t.msno = l.msno
               )
               SELECT user_id, count(*) AS raw_rows,
                      count(*) FILTER (WHERE user_id IS NOT NULL) AS matched_rows,
                      count(*) FILTER (WHERE user_id IS NOT NULL AND event_date IS NULL) AS invalid_dates,
                      count(*) FILTER (WHERE user_id IS NOT NULL AND event_date > CAST(? AS DATE)) AS future_rows,
                      count(*) FILTER (WHERE user_id IS NOT NULL AND event_date <= CAST(? AS DATE)) AS usable_rows,
                      sum(TRY_CAST(coalesce(nullif(actual_amount_paid, ''), '0') AS DOUBLE))
                          FILTER (WHERE user_id IS NOT NULL AND event_date <= CAST(? AS DATE)) AS total_paid,
                      arg_max(
                          struct_pack(
                              plan := TRY_CAST(coalesce(nullif(payment_plan_days, ''), '0') AS INTEGER),
                              renew := TRY_CAST(coalesce(nullif(is_auto_renew, ''), '0') AS INTEGER)
                          ),
                          struct_pack(
                              event := event_date,
                              plan := TRY_CAST(coalesce(nullif(payment_plan_days, ''), '0') AS INTEGER),
                              renew := TRY_CAST(coalesce(nullif(is_auto_renew, ''), '0') AS INTEGER)
                          )
                      ) FILTER (WHERE user_id IS NOT NULL AND event_date <= CAST(? AS DATE)) AS latest_plan,
                      max(event_date) FILTER (WHERE user_id IS NOT NULL AND event_date <= CAST(? AS DATE)) AS latest_date,
                      count(*) FILTER (WHERE user_id IS NOT NULL AND event_date <= CAST(? AS DATE)
                          AND (TRY_CAST(coalesce(nullif(actual_amount_paid, ''), '0') AS DOUBLE) IS NULL
                            OR TRY_CAST(coalesce(nullif(payment_plan_days, ''), '0') AS INTEGER) IS NULL
                            OR TRY_CAST(coalesce(nullif(is_auto_renew, ''), '0') AS INTEGER) IS NULL)) AS invalid_values
               FROM parsed GROUP BY user_id""",
            [str(transactions_path), *([cutoff] * 6)],
        )
        tx_counts = db.execute(
            """SELECT sum(raw_rows), sum(matched_rows), sum(future_rows), sum(invalid_dates),
                      sum(usable_rows), sum(invalid_values) FROM tx_agg"""
        ).fetchone()
        if tx_counts[5]:
            raise ValueError(f"Transactions contain {tx_counts[5]} invalid pre-snapshot numeric rows")

        log_counts = (0, 0, 0, 0, 0, 0)
        if logs_path is not None:
            print("Scanning listening logs with DuckDB", flush=True)
            db.execute(
                """CREATE OR REPLACE TABLE log_agg AS
                   WITH parsed AS (
                       SELECT l.msno AS user_id,
                              TRY_STRPTIME(r.date, '%Y%m%d')::DATE AS event_date,
                              r.total_secs, r.num_unq
                       FROM read_csv(?, header=true, all_varchar=true) AS r
                       LEFT JOIN labels AS l ON r.msno = l.msno
                   )
                   SELECT user_id, count(*) AS raw_rows,
                          count(*) FILTER (WHERE user_id IS NOT NULL) AS matched_rows,
                          count(*) FILTER (WHERE user_id IS NOT NULL AND event_date IS NULL) AS invalid_dates,
                          count(*) FILTER (WHERE user_id IS NOT NULL AND event_date > CAST(? AS DATE)) AS future_rows,
                          count(*) FILTER (WHERE user_id IS NOT NULL AND event_date <= CAST(? AS DATE)) AS usable_rows,
                          sum(TRY_CAST(coalesce(nullif(total_secs, ''), '0') AS DOUBLE))
                              FILTER (WHERE user_id IS NOT NULL AND event_date <= CAST(? AS DATE)) AS total_seconds,
                          sum(TRY_CAST(coalesce(nullif(num_unq, ''), '0') AS BIGINT))
                              FILTER (WHERE user_id IS NOT NULL AND event_date <= CAST(? AS DATE)) AS unique_tracks,
                          max(event_date) FILTER (WHERE user_id IS NOT NULL AND event_date <= CAST(? AS DATE)) AS latest_date,
                          count(*) FILTER (WHERE user_id IS NOT NULL AND event_date <= CAST(? AS DATE)
                              AND (TRY_CAST(coalesce(nullif(total_secs, ''), '0') AS DOUBLE) IS NULL
                                OR TRY_CAST(coalesce(nullif(num_unq, ''), '0') AS BIGINT) IS NULL)) AS invalid_values
                   FROM parsed GROUP BY user_id""",
                [str(logs_path), *([cutoff] * 6)],
            )
            log_counts = db.execute(
                """SELECT sum(raw_rows), sum(matched_rows), sum(future_rows), sum(invalid_dates),
                          sum(usable_rows), sum(invalid_values) FROM log_agg"""
            ).fetchone()
            if log_counts[5]:
                raise ValueError(f"Listening logs contain {log_counts[5]} invalid pre-snapshot numeric rows")

        if require_coverage and (not tx_counts[4] or (logs_path is not None and not log_counts[4])):
            raise ValueError("No pre-snapshot transactions or listening logs matched the labeled users")

        log_join = "LEFT JOIN log_agg AS g ON l.msno = g.user_id" if logs_path else ""
        log_select = (
            "coalesce(g.usable_rows, 0) AS log_rows, "
            "coalesce(g.total_seconds, 0) AS total_listening_seconds, "
            "coalesce(g.unique_tracks, 0) AS total_unique_tracks, "
            "coalesce(date_diff('day', g.latest_date, CAST(? AS DATE)), -1) AS last_log_days"
            if logs_path else
            "0 AS log_rows, 0.0 AS total_listening_seconds, 0 AS total_unique_tracks, -1 AS last_log_days"
        )
        parameters = [cutoff, cutoff] if logs_path else [cutoff]
        features = db.execute(
            f"""SELECT l.msno, l.is_churn,
                       coalesce(t.usable_rows, 0) AS transaction_count,
                       coalesce(t.total_paid, 0) AS total_paid,
                       coalesce(date_diff('day', t.latest_date, CAST(? AS DATE)), -1) AS last_transaction_days,
                       coalesce(t.latest_plan.plan, 0) AS last_plan_days,
                       coalesce(t.latest_plan.renew, 0) AS last_auto_renew,
                       {log_select}
                FROM labels AS l LEFT JOIN tx_agg AS t ON l.msno = t.user_id
                {log_join} ORDER BY l.msno""",
            parameters,
        ).df()
        audit = {
            "transaction_rows_read": int(tx_counts[0]),
            "transaction_rows_for_labeled_users": int(tx_counts[1]),
            "transaction_rows_after_snapshot": int(tx_counts[2]),
            "transaction_rows_invalid_date": int(tx_counts[3]),
            "log_rows_read": int(log_counts[0]),
            "log_rows_for_labeled_users": int(log_counts[1]),
            "log_rows_after_snapshot": int(log_counts[2]),
            "log_rows_invalid_date": int(log_counts[3]),
            "label_rows": int(label_rows),
            "churn_labels": int(churn_labels),
            "pre_snapshot_transaction_rows": int(tx_counts[4]),
            "pre_snapshot_log_rows": int(log_counts[4]),
            "users_without_transactions": int((features["transaction_count"] == 0).sum()),
            "users_without_logs": int((features["log_rows"] == 0).sum()),
        }
        return features, audit
