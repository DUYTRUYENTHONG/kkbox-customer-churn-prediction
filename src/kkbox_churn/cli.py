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
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--logs", type=Path, help="user_logs.csv (pre-cutoff history)")
    sources.add_argument("--transactions-only", action="store_true", help="run without listening history")
    parser.add_argument("--snapshot", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--engine", choices=("python", "duckdb"), default="python")
    parser.add_argument("--audit-only", action="store_true", help="Report input coverage without training")
    args = parser.parse_args()

    if args.engine == "duckdb":
        from .duckdb_features import build_features_with_audit_duckdb

        data, audit = build_features_with_audit_duckdb(
            args.labels, args.transactions, args.logs, args.snapshot, args.output,
            require_coverage=not args.audit_only,
        )
    else:
        data, audit = build_features_with_audit(
            args.labels, args.transactions, args.logs, args.snapshot,
            require_coverage=not args.audit_only,
        )
    report = {
        "engine": args.engine,
        "feature_mode": "transactions_and_logs" if args.logs else "transactions_only",
        "snapshot": args.snapshot.date().isoformat(),
        "label_file": args.labels.name,
        "transaction_file": args.transactions.name,
        "log_file": args.logs.name if args.logs else None,
        **audit,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if args.transactions_only:
        data = data.drop(columns=["log_rows", "total_listening_seconds", "total_unique_tracks", "last_log_days"])
    metrics = train_baseline(data, args.output / "baseline.joblib")
    metrics.update(report)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
