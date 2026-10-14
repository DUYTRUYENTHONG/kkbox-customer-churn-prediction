import csv
from datetime import datetime

from kkbox_churn.features import build_features


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_features_exclude_events_after_snapshot(tmp_path):
    labels = tmp_path / "train.csv"
    members = tmp_path / "members.csv"
    transactions = tmp_path / "transactions.csv"
    logs = tmp_path / "logs.csv"
    write_csv(labels, [{"msno": "a", "is_churn": "1"}])
    write_csv(members, [{"msno": "a", "city": "1", "registered_via": "7"}])
    write_csv(transactions, [
        {"msno": "a", "transaction_date": "20170227", "actual_amount_paid": "100", "payment_plan_days": "30", "is_auto_renew": "1"},
        {"msno": "a", "transaction_date": "20170302", "actual_amount_paid": "999", "payment_plan_days": "90", "is_auto_renew": "0"},
    ])
    write_csv(logs, [
        {"msno": "a", "date": "20170226", "total_secs": "50", "num_unq": "2"},
        {"msno": "a", "date": "20170301", "total_secs": "500", "num_unq": "20"},
    ])
    result = build_features(labels, members, transactions, logs, datetime(2017, 2, 28)).iloc[0]
    assert result["transaction_count"] == 1
    assert result["total_paid"] == 100
    assert result["last_transaction_days"] == 1
    assert result["log_days"] == 1
    assert result["total_listening_seconds"] == 50
