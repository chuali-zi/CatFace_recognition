import json
import os
from pathlib import Path
import subprocess
import sys


def test_evaluation_cli_runs_outside_repo(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        dict(sample_id="1", split="calibration_unknown", cat_id="x", predicted_id="a", score=0.4),
        dict(sample_id="2", split="known", cat_id="a", predicted_id="a", score=0.9),
        dict(sample_id="3", split="unknown", cat_id="y", predicted_id="a", score=0.3),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "metrics.json"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-m", "catid", "evaluate", str(path), "--out", str(output)],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["dir"] == 1
    assert metrics["fpir"] == 0
