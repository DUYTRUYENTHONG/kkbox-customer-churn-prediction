"""Exercise the public command with synthetic files, never competition data."""

import csv
import json
import sys

from kkbox_churn.cli import main


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_cli_writes_model_metrics_and_aggregate_audit(tmp_path, monkeypatch):
    labels = tmp_path / "train_v2.csv"
    transactions = tmp_path / "transactions.csv"
    logs = tmp_path / "user_logs.csv"
    output = tmp_path / "output"
    _write_csv(labels, [
        {"msno": f"user-{index}", "is_churn": index % 2}
        for index in range(40)
    ])
    _write_csv(transactions, [
        {
            "msno": f"user-{index}",
            "transaction_date": "20170227",
            "actual_amount_paid": 50 + index,
            "payment_plan_days": 30,
            "is_auto_renew": index % 2,
        }
        for index in range(40)
    ])
    _write_csv(logs, [
        {"msno": f"user-{index}", "date": "20170227", "total_secs": 100 + index, "num_unq": 3}
        for index in range(40)
    ])
    monkeypatch.setattr(sys, "argv", [
        "kkbox-churn", "--labels", str(labels), "--transactions", str(transactions),
        "--logs", str(logs), "--snapshot", "2017-02-28", "--output", str(output),
    ])

    main()

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert (output / "baseline.joblib").is_file()
    assert metrics["users"] == 40
    assert metrics["pre_snapshot_transaction_rows"] == 40
    assert metrics["pre_snapshot_log_rows"] == 40
    assert metrics["transaction_rows_after_snapshot"] == 0
    assert metrics["log_rows_invalid_date"] == 0
    assert metrics["snapshot"] == "2017-02-28"

    audit_output = tmp_path / "audit-only"
    monkeypatch.setattr(sys, "argv", [
        "kkbox-churn", "--labels", str(labels), "--transactions", str(transactions),
        "--logs", str(logs), "--snapshot", "2017-02-28",
        "--audit-only", "--output", str(audit_output),
    ])
    main()
    audit = json.loads((audit_output / "audit.json").read_text(encoding="utf-8"))
    assert audit["pre_snapshot_transaction_rows"] == 40
    assert not (audit_output / "baseline.joblib").exists()
    assert not (audit_output / "metrics.json").exists()


def test_cli_transactions_only_omits_log_features(tmp_path, monkeypatch):
    labels = tmp_path / "train_v2.csv"
    transactions = tmp_path / "transactions.csv"
    output = tmp_path / "output"
    _write_csv(labels, [
        {"msno": f"user-{index}", "is_churn": index % 2}
        for index in range(40)
    ])
    _write_csv(transactions, [
        {
            "msno": f"user-{index}",
            "transaction_date": "20170227",
            "actual_amount_paid": 50 + index,
            "payment_plan_days": 30,
            "is_auto_renew": index % 2,
        }
        for index in range(40)
    ])
    monkeypatch.setattr(sys, "argv", [
        "kkbox-churn", "--labels", str(labels), "--transactions", str(transactions),
        "--transactions-only", "--snapshot", "2017-02-28", "--output", str(output),
    ])
    main()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["feature_mode"] == "transactions_only"
    assert metrics["log_file"] is None
    assert metrics["pre_snapshot_log_rows"] == 0
    assert metrics["pre_snapshot_transaction_rows"] == 40
    assert (output / "baseline.joblib").is_file()

    fast_output = tmp_path / "fast-output"
    monkeypatch.setattr(sys, "argv", [
        "kkbox-churn", "--labels", str(labels), "--transactions", str(transactions),
        "--transactions-only", "--snapshot", "2017-02-28", "--engine", "duckdb",
        "--output", str(fast_output),
    ])
    main()
    fast_metrics = json.loads((fast_output / "metrics.json").read_text(encoding="utf-8"))
    assert fast_metrics["engine"] == "duckdb"
    assert fast_metrics["pre_snapshot_transaction_rows"] == 40
