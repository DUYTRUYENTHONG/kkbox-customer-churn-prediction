"""Build customer features using only events known by a snapshot date."""

from __future__ import annotations

import csv
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


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def build_features(
    labels_path: Path,
    transactions_path: Path,
    logs_path: Path,
    snapshot: datetime,
) -> pd.DataFrame:
    """Return one row per labeled user; future events never enter features."""
    labels: dict[str, int] = {}
    for row in _rows(labels_path):
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
            "log_days": 0,
            "total_listening_seconds": 0.0,
            "total_unique_tracks": 0,
            "last_log_days": -1,
        }
        for key, label in labels.items()
    }

    latest_tx: dict[str, datetime] = {}
    accepted_transactions = 0
    for row in _rows(transactions_path):
        key = row["msno"]
        user = users.get(key)
        event_date = _date(row.get("transaction_date", ""))
        if user is None or event_date is None or event_date > snapshot:
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
    for row in _rows(logs_path):
        key = row["msno"]
        user = users.get(key)
        event_date = _date(row.get("date", ""))
        if user is None or event_date is None or event_date > snapshot:
            continue
        accepted_logs += 1
        user["log_days"] += 1
        user["total_listening_seconds"] += float(row.get("total_secs") or 0)
        user["total_unique_tracks"] += int(row.get("num_unq") or 0)
        if key not in latest_log or event_date >= latest_log[key]:
            latest_log[key] = event_date
            user["last_log_days"] = (snapshot - event_date).days

    if not accepted_transactions or not accepted_logs:
        raise ValueError(
            "No pre-snapshot transactions or listening logs matched the labeled users. "
            "For a 2017-02-28 cutoff, use transactions.csv and user_logs.csv; "
            "the _v2 event files contain March data."
        )

    return pd.DataFrame(users.values()).sort_values("msno").reset_index(drop=True)
