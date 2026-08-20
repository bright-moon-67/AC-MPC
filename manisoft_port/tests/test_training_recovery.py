import json

from scripts.train_koopman import reconcile_history


def test_reconcile_history_truncates_future_and_deduplicates(tmp_path):
    history = tmp_path / "history.jsonl"
    rows = [
        {"epoch": 0, "validation": {"total": 3.0}},
        {"epoch": 1, "validation": {"total": 2.0}},
        {"epoch": 2, "validation": {"total": 1.5}},
        {"epoch": 2, "validation": {"total": 1.0}},
        {"epoch": 3, "validation": {"total": 0.5}},
    ]
    history.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    best_epoch, removed = reconcile_history(history, start_epoch=3)
    reconciled = [
        json.loads(line)
        for line in history.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["epoch"] for row in reconciled] == [0, 1, 2]
    assert reconciled[-1]["validation"]["total"] == 1.0
    assert best_epoch == 2
    assert removed == 2
