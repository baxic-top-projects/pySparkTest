"""MLflow: tracking URI из Docker и хелперы для ноутбука."""
from __future__ import annotations

import os

# В Docker-образе Jupyter нет git — MLflow пытается записать commit SHA и шумит в лог.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from typing import Any

import mlflow
import numpy as np
from mlflow.models import infer_signature

SPELL_FEATURE_NAMES = ("mana_cost", "damage")
MLFLOW_ARTIFACT_ROOT = "mlflow-artifacts:/"


def get_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def mlflow_ui_url() -> str:
    """URL для браузера с хоста (Jupyter в Docker → localhost)."""
    return get_tracking_uri().replace("http://mlflow:5000", "http://localhost:5000")


def _experiment_needs_recreate(artifact_location: str | None) -> bool:
    """Пути вида /mlflow/artifacts/N — только на сервере, Jupyter пишет локально и падает."""
    if not artifact_location:
        return False
    loc = artifact_location.strip()
    if loc.startswith("mlflow-artifacts:"):
        return False
    if loc.startswith("file:") or "/mlflow/artifacts" in loc:
        return True
    return loc.startswith("/mlflow")


def _ensure_experiment(client: Any, experiment_name: str) -> str:
    from mlflow.exceptions import RestException

    exp = client.get_experiment_by_name(experiment_name)
    if exp is not None:
        stage = getattr(exp, "lifecycle_stage", "active")
        if stage == "deleted":
            raise RuntimeError(
                f"Experiment '{experiment_name}' в корзине MLflow (имя занято).\n"
                "Выполните: docker compose restart mlflow\n"
                "или: docker compose exec mlflow sqlite3 /mlflow/mlflow.db "
                f"\"DELETE FROM experiments WHERE name='{experiment_name}';\""
            )
        if _experiment_needs_recreate(exp.artifact_location):
            print(
                f"Удаляем experiment '{experiment_name}' "
                f"(artifact_location={exp.artifact_location!r}) — runs будут удалены."
            )
            client.delete_experiment(exp.experiment_id)
            exp = None

    if exp is None:
        try:
            return client.create_experiment(
                experiment_name,
                artifact_location=MLFLOW_ARTIFACT_ROOT,
            )
        except RestException as exc:
            if "UNIQUE" in str(exc):
                raise RuntimeError(
                    f"Имя '{experiment_name}' занято в БД MLflow.\n"
                    "Выполните: docker compose restart mlflow"
                ) from exc
            raise

    return exp.experiment_id


def setup_mlflow(experiment_name: str = "spells_classifiers") -> str:
    """Подключение к MLflow Server и выбор experiment."""
    from mlflow.tracking import MlflowClient

    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    exp_id = _ensure_experiment(client, experiment_name)
    mlflow.set_experiment(experiment_id=exp_id)

    exp = client.get_experiment(exp_id)
    print(f"artifact_location: {exp.artifact_location}")
    print(f"MLflow tracking: {uri}")
    print(f"UI в браузере: {mlflow_ui_url()}")
    print(f"Experiment: {experiment_name}")
    return uri


def log_sklearn_spell_model(
    sk_model: Any,
    *,
    classifier_name: str,
    X_sample: np.ndarray,
    metrics: dict[str, float] | None = None,
    artifact_path: str = "model",
) -> None:
    """Сохраняет sklearn-модель в текущий MLflow run (артефакт + signature)."""
    X_ex = np.asarray(X_sample[:5])
    signature = infer_signature(X_ex, sk_model.predict(X_ex))

    for key, value in (metrics or {}).items():
        mlflow.log_metric(key, float(value))

    mlflow.set_tag("classifier", classifier_name)
    mlflow.set_tag("task", "spell_fireball_vs_lightning")
    mlflow.log_param("features", ",".join(SPELL_FEATURE_NAMES))

    mlflow.sklearn.log_model(
        sk_model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=X_ex[:1],
    )
    print(f"  → модель сохранена в MLflow: artifact '{artifact_path}'")
