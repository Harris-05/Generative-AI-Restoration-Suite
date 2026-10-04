"""
Experiment tracking with MLflow, shared by all four tasks.

Every training run is recorded with its hyperparameters, per-epoch metrics, and
the checkpoint it produced. All runs live in one local store at
<project>/mlflow/, so you can inspect them with:

  mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

If MLflow is not installed, the tracker prints a warning and does nothing, so
training is never blocked by tracking.
"""

from pathlib import Path

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = PROJECT_ROOT / "mlflow"


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


class Tracker:
    def __init__(self, experiment: str, run_name: str, params: dict):
        self.active = False
        if mlflow is None:
            print("MLflow is not installed; experiment tracking is disabled "
                  "(pip install mlflow).")
            return
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        artifact_dir = STORE_DIR / "artifacts" / experiment
        mlflow.set_tracking_uri(f"sqlite:///{STORE_DIR / 'mlflow.db'}")
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=artifact_dir.as_uri())
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({k: str(v) for k, v in _flatten(params).items()})
        self.active = True

    def log_metrics(self, metrics: dict, step: int):
        if self.active:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: Path):
        if self.active and Path(path).exists():
            mlflow.log_artifact(str(path))

    def close(self):
        if self.active:
            mlflow.end_run()
            self.active = False
