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
    parser.add_argument("--members", type=Path, required=True, help="members_v3.csv")
    parser.add_argument("--transactions", type=Path, required=True, help="transactions_v2.csv")
    parser.add_argument("--logs", type=Path, required=True, help="user_logs_v2.csv")
    parser.add_argument("--snapshot", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    data = build_features(args.labels, args.members, args.transactions, args.logs, args.snapshot)
    metrics = train_baseline(data, args.output / "baseline.joblib")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
