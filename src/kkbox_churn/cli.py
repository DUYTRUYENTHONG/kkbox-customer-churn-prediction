"""Command-line entrypoint for a local KKBox baseline experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .features import build_features_with_audit
from .model import train_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an as-of-date KKBox churn baseline")
    parser.add_argument("--labels", type=Path, required=True, help="CSV with msno,is_churn")
    parser.add_argument("--transactions", type=Path, required=True, help="transactions.csv (pre-cutoff history)")
    parser.add_argument("--logs", type=Path, required=True, help="user_logs.csv (pre-cutoff history)")
    parser.add_argument("--snapshot", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--audit-only", action="store_true", help="Report input coverage without training")
    args = parser.parse_args()

    data, audit = build_features_with_audit(
        args.labels, args.transactions, args.logs, args.snapshot,
        require_coverage=not args.audit_only,
    )
    report = {
        "snapshot": args.snapshot.date().isoformat(),
        "label_file": args.labels.name,
        "transaction_file": args.transactions.name,
        "log_file": args.logs.name,
        **audit,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    metrics = train_baseline(data, args.output / "baseline.joblib")
    metrics.update(report)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
