import numpy as np
import pandas as pd

from kaggle_multi_agent.data import CompetitionFrames
from kaggle_multi_agent.graph import run_agent_loop
from kaggle_multi_agent.rag import LocalKnowledgeBase


def test_run_agent_loop_completes_with_best_result(tmp_path) -> None:
    source_dir = tmp_path / "curated"
    source_dir.mkdir()
    (source_dir / "notes.md").write_text(
        "Tune tree depth carefully. Use offline metrics to compare runs.",
        encoding="utf-8",
    )
    kb = LocalKnowledgeBase.build(source_dir)
    rows = 40
    train = pd.DataFrame(
        {
            "name": [f"name-{idx % 5}" for idx in range(rows)],
            "_id": list(range(rows)),
            "host_name": [f"host-{idx % 4}" for idx in range(rows)],
            "location_cluster": ["A" if idx % 2 == 0 else "B" for idx in range(rows)],
            "location": [f"loc-{idx % 3}" for idx in range(rows)],
            "lat": np.linspace(1.0, 4.0, rows),
            "lon": np.linspace(10.0, 12.0, rows),
            "type_house": ["flat" if idx % 2 == 0 else "room" for idx in range(rows)],
            "sum": np.linspace(50, 150, rows),
            "min_days": [1 + idx % 5 for idx in range(rows)],
            "amt_reviews": [idx % 7 for idx in range(rows)],
            "last_dt": ["2019-01-02"] * rows,
            "avg_reviews": np.linspace(0.1, 3.0, rows),
            "total_host": [1 + idx % 3 for idx in range(rows)],
            "target": [idx % 30 for idx in range(rows)],
        }
    )
    test = train.drop(columns=["target"]).head(8).copy()
    solution = pd.DataFrame(
        {
            "index": list(range(len(test))),
            "prediction": [idx % 10 for idx in range(len(test))],
            "Usage": ["Public"] * 4 + ["Private"] * 4,
        }
    )
    sample = pd.DataFrame({"index": list(range(len(test))), "prediction": [0] * len(test)})
    frames = CompetitionFrames(
        train=train,
        test=test,
        sample_submission=sample,
        solution=solution,
    )
    final_state = run_agent_loop(
        frames=frames,
        knowledge_base=kb,
        target_column="target",
        max_iterations=2,
        random_seed=42,
    )
    assert final_state["best_result"] is not None
    assert len(final_state["history"]) >= 1
    assert final_state["best_spec"] is not None
    assert final_state["tool_trace_history"]
