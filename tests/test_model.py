import pandas as pd

from kkbox_churn.model import train_baseline


def test_model_trains_and_records_holdout_metrics(tmp_path):
    rows = []
    for index in range(40):
        rows.append({
            "msno": f"user-{index}",
            "is_churn": index % 2,
            "transaction_count": 1 + index % 4,
            "total_paid": float(50 + index * 3),
            "last_transaction_days": index % 15,
            "last_plan_days": 30,
            "last_auto_renew": index % 2,
            "log_rows": 2 + index % 6,
            "total_listening_seconds": float(index * 20),
            "total_unique_tracks": index % 9,
            "last_log_days": index % 11,
        })
    path = tmp_path / "model.joblib"
    metrics = train_baseline(pd.DataFrame(rows), path)
    assert path.is_file()
    assert metrics["users"] == 40
    assert metrics["holdout_users"] == 10
    assert 0 <= metrics["holdout_auc_roc"] <= 1
    assert metrics["holdout_log_loss"] >= 0
    assert metrics["prevalence_log_loss"] >= 0
    assert metrics["top_decile_users"] == 1
