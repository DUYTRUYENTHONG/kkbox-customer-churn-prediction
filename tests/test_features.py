import csv
from datetime import datetime

import pytest

from kkbox_churn.features import build_features


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_features_exclude_events_after_snapshot(tmp_path):
    labels = tmp_path / "train.csv"
    transactions = tmp_path / "transactions.csv"
    logs = tmp_path / "logs.csv"
    write_csv(labels, [{"msno": "a", "is_churn": "1"}])
    write_csv(transactions, [
        {"msno": "a", "transaction_date": "20170227", "actual_amount_paid": "100", "payment_plan_days": "30", "is_auto_renew": "1"},
        {"msno": "a", "transaction_date": "20170302", "actual_amount_paid": "999", "payment_plan_days": "90", "is_auto_renew": "0"},
    ])
    write_csv(logs, [
        {"msno": "a", "date": "20170226", "total_secs": "50", "num_unq": "2"},
        {"msno": "a", "date": "20170301", "total_secs": "500", "num_unq": "20"},
    ])
    result = build_features(labels, transactions, logs, datetime(2017, 2, 28)).iloc[0]
    assert result["transaction_count"] == 1
    assert result["total_paid"] == 100
    assert result["last_transaction_days"] == 1
    assert result["log_days"] == 1
    assert result["total_listening_seconds"] == 50


def test_rejects_event_files_with_only_post_snapshot_rows(tmp_path):
    labels = tmp_path / "train_v2.csv"
    transactions = tmp_path / "transactions_v2.csv"
    logs = tmp_path / "user_logs_v2.csv"
    write_csv(labels, [{"msno": "a", "is_churn": "1"}])
    write_csv(transactions, [{"msno": "a", "transaction_date": "20170301", "actual_amount_paid": "100", "payment_plan_days": "30", "is_auto_renew": "1"}])
    write_csv(logs, [{"msno": "a", "date": "20170301", "total_secs": "50", "num_unq": "2"}])

    with pytest.raises(ValueError, match="No pre-snapshot transactions or listening logs"):
        build_features(labels, transactions, logs, datetime(2017, 2, 28))


def test_rejects_duplicate_label_rows(tmp_path):
    labels = tmp_path / "train_v2.csv"
    write_csv(labels, [{"msno": "a", "is_churn": "1"}, {"msno": "a", "is_churn": "0"}])
    with pytest.raises(ValueError, match="one nonempty row per msno"):
        build_features(labels, tmp_path / "unused-transactions.csv", tmp_path / "unused-logs.csv", datetime(2017, 2, 28))
