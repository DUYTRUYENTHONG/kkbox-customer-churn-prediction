"""Command-line entrypoint for a local KKBox baseline experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .features import build_features
from .model import train_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an as-of-date KKBox churn baseline")
    parser.add_argument("--labels", type=Path, required=True, help="CSV with msno,is_churn")
    parser.add_argument("--transactions", type=Path, required=True, help="transactions.csv (pre-cutoff history)")
    parser.add_argument("--logs", type=Path, required=True, help="user_logs.csv (pre-cutoff history)")
    parser.add_argument("--snapshot", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    data = build_features(args.labels, args.transactions, args.logs, args.snapshot)
    metrics = train_baseline(data, args.output / "baseline.joblib")
    metrics.update({
        "snapshot": args.snapshot.date().isoformat(),
        "label_file": args.labels.name,
        "transaction_file": args.transactions.name,
        "log_file": args.logs.name,
        "pre_snapshot_transaction_rows": int(data["transaction_count"].sum()),
        "pre_snapshot_log_rows": int(data["log_days"].sum()),
        "users_without_transactions": int((data["transaction_count"] == 0).sum()),
        "users_without_logs": int((data["log_days"] == 0).sum()),
    })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
