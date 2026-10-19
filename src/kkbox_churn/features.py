"""Build customer features using only events known by a snapshot date."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def _date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return None


def _rows(path: Path, required: set[str]):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path.name} has duplicate CSV column names")
        missing = required - set(columns)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name} has a row whose field count differs from its header")
            yield row


def build_features(
    labels_path: Path,
    transactions_path: Path,
    logs_path: Path,
    snapshot: datetime,
) -> pd.DataFrame:
    """Return one row per labeled user; future events never enter features."""
    features, _ = build_features_with_audit(labels_path, transactions_path, logs_path, snapshot)
    return features


def build_features_with_audit(
    labels_path: Path,
    transactions_path: Path,
    logs_path: Path | None,
    snapshot: datetime,
    *,
    require_coverage: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build features and aggregate input diagnostics without exposing customer IDs."""
    audit = {
        "transaction_rows_read": 0,
        "transaction_rows_for_labeled_users": 0,
        "transaction_rows_after_snapshot": 0,
        "transaction_rows_invalid_date": 0,
        "log_rows_read": 0,
        "log_rows_for_labeled_users": 0,
        "log_rows_after_snapshot": 0,
        "log_rows_invalid_date": 0,
    }
    labels: dict[str, int] = {}
    for row in _rows(labels_path, {"msno", "is_churn"}):
        key = row["msno"]
        if not key or key in labels:
            raise ValueError("Labels must have one nonempty row per msno")
        label = int(row["is_churn"])
        if label not in (0, 1):
            raise ValueError("is_churn must be 0 or 1")
        labels[key] = label
    if not labels:
        raise ValueError("No labeled users were found")
    users: dict[str, dict[str, float | int | str]] = {
        key: {
            "msno": key,
            "is_churn": label,
            "transaction_count": 0,
            "total_paid": 0.0,
            "last_transaction_days": -1,
            "last_plan_days": 0,
            "last_auto_renew": 0,
            "log_rows": 0,
            "total_listening_seconds": 0.0,
            "total_unique_tracks": 0,
            "last_log_days": -1,
        }
        for key, label in labels.items()
    }

    latest_tx: dict[str, datetime] = {}
    accepted_transactions = 0
    next_tx_progress = 10_000_000
    for row in _rows(
        transactions_path,
        {"msno", "transaction_date", "actual_amount_paid", "payment_plan_days", "is_auto_renew"},
    ):
        audit["transaction_rows_read"] += 1
        if audit["transaction_rows_read"] >= next_tx_progress:
            print(f"Read {next_tx_progress:,} transaction rows", file=sys.stderr, flush=True)
            next_tx_progress += 10_000_000
        key = row["msno"]
        user = users.get(key)
        if user is None:
            continue
        audit["transaction_rows_for_labeled_users"] += 1
        event_date = _date(row.get("transaction_date", ""))
        if event_date is None:
            audit["transaction_rows_invalid_date"] += 1
            continue
        if event_date > snapshot:
            audit["transaction_rows_after_snapshot"] += 1
            continue
        accepted_transactions += 1
        user["transaction_count"] += 1
        user["total_paid"] += float(row.get("actual_amount_paid") or 0)
        if key not in latest_tx or event_date >= latest_tx[key]:
            latest_tx[key] = event_date
            user["last_transaction_days"] = (snapshot - event_date).days
            user["last_plan_days"] = int(row.get("payment_plan_days") or 0)
            user["last_auto_renew"] = int(row.get("is_auto_renew") or 0)

    latest_log: dict[str, datetime] = {}
    accepted_logs = 0
    next_log_progress = 10_000_000
    log_rows = _rows(logs_path, {"msno", "date", "total_secs", "num_unq"}) if logs_path else ()
    for row in log_rows:
        audit["log_rows_read"] += 1
        if audit["log_rows_read"] >= next_log_progress:
            print(f"Read {next_log_progress:,} listening rows", file=sys.stderr, flush=True)
            next_log_progress += 10_000_000
        key = row["msno"]
        user = users.get(key)
        if user is None:
            continue
        audit["log_rows_for_labeled_users"] += 1
        event_date = _date(row.get("date", ""))
        if event_date is None:
            audit["log_rows_invalid_date"] += 1
            continue
        if event_date > snapshot:
            audit["log_rows_after_snapshot"] += 1
            continue
        accepted_logs += 1
        user["log_rows"] += 1
        user["total_listening_seconds"] += float(row.get("total_secs") or 0)
        user["total_unique_tracks"] += int(row.get("num_unq") or 0)
        if key not in latest_log or event_date >= latest_log[key]:
            latest_log[key] = event_date
            user["last_log_days"] = (snapshot - event_date).days

    if require_coverage and (not accepted_transactions or (logs_path and not accepted_logs)):
        sources = "transactions or listening logs" if logs_path else "transactions"
        raise ValueError(
            f"No pre-snapshot {sources} matched the labeled users. "
            "For a 2017-02-28 cutoff, use transactions.csv and user_logs.csv; "
            "the _v2 event files are March updates."
        )

    audit.update(
        {
            "label_rows": len(labels),
            "churn_labels": sum(labels.values()),
            "pre_snapshot_transaction_rows": accepted_transactions,
            "pre_snapshot_log_rows": accepted_logs,
            "users_without_transactions": sum(
                user["transaction_count"] == 0 for user in users.values()
            ),
            "users_without_logs": sum(user["log_rows"] == 0 for user in users.values()),
        }
    )
    return pd.DataFrame(users.values()).sort_values("msno").reset_index(drop=True), audit
