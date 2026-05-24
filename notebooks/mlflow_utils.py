"""MLflow: tracking URI из Docker и хелперы для ноутбука."""
from __future__ import annotations

import os

# В Docker-образе Jupyter нет git — MLflow пытается записать commit SHA и шумит в лог.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from typing import Any

import mlflow
import numpy as np
from mlflow.models import infer_signature

__version__ = "2"  # bump при изменении API (для importlib.reload в ноутбуке)

SPELL_FEATURE_NAMES = ("mana_cost", "damage")
MLFLOW_ARTIFACT_ROOT = "mlflow-artifacts:/"


def get_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def mlflow_ui_url() -> str:
    """URL для браузера с хоста (Jupyter в Docker → localhost)."""
    return get_tracking_uri().replace("http://mlflow:5000", "http://localhost:5000")


def get_serve_uri() -> str:
    return os.environ.get("MLFLOW_SERVE_URI", "http://mlflow-serve:5001")


def predict_spells_via_serve(
    rows: list[list[float]],
    *,
    columns: tuple[str, ...] = SPELL_FEATURE_NAMES,
    serve_uri: str | None = None,
    timeout: float = 30.0,
) -> Any:
    """Предсказание через MLflow Model Server (POST /invocations), без загрузки модели в память."""
    import requests

    uri = (serve_uri or get_serve_uri()).rstrip("/")
    payload = {"dataframe_split": {"columns": list(columns), "data": rows}}
    resp = requests.post(f"{uri}/invocations", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


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


REGISTERED_MODEL_NAME = "spells-classifier"
COMPARE_METRICS = ("test_f1", "test_roc_auc", "accuracy_real", "holdout_f1")


def list_classifier_runs(
    experiment_name: str = "spells_classifiers",
    *,
    metric_names: tuple[str, ...] = COMPARE_METRICS,
) -> "pd.DataFrame":
    """Таблица runs experiment с метриками (для сравнения моделей в UI/ноутбуке)."""
    import pandas as pd
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' не найден. Сначала обучите модели.")

    rows: list[dict[str, Any]] = []
    for run in client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    ):
        row: dict[str, Any] = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "classifier": run.data.params.get("classifier") or run.info.run_name,
        }
        for key in metric_names:
            row[key] = run.data.metrics.get(key)
        rows.append(row)

    return pd.DataFrame(rows)


def select_best_run(
    runs_df: "pd.DataFrame",
    *,
    primary_metric: str = "test_f1",
    fallback_metrics: tuple[str, ...] = ("test_roc_auc", "accuracy_real"),
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Выбирает лучший run по primary_metric, затем по fallback."""
    if runs_df.empty:
        raise RuntimeError("Нет завершённых runs в experiment.")

    metrics_to_try = (primary_metric,) + fallback_metrics
    last_error: str | None = None

    for metric in metrics_to_try:
        if metric not in runs_df.columns:
            continue
        subset = runs_df.dropna(subset=[metric])
        if subset.empty:
            last_error = f"ни у одного run нет метрики '{metric}'"
            continue
        best_idx = (
            subset[metric].idxmax() if higher_is_better else subset[metric].idxmin()
        )
        row = subset.loc[best_idx]
        return {
            "run_id": row["run_id"],
            "classifier": row["classifier"],
            "metric": metric,
            "value": float(row[metric]),
        }

    raise RuntimeError(
        f"Не удалось выбрать лучший run: {last_error or 'нет метрик'}. "
        f"Выполните ячейку с test_f1 / accuracy_real."
    )


def register_and_promote_run(
    run_id: str,
    *,
    model_name: str = REGISTERED_MODEL_NAME,
    target_stage: str = "Staging",
    artifact_path: str = "model",
) -> Any:
    """Регистрирует модель из run и переводит версию на следующий этап (Staging/Production)."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    version_info = mlflow.register_model(model_uri, model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=version_info.version,
        stage=target_stage,
        archive_existing_versions=(target_stage == "Production"),
    )
    print(
        f"Model Registry: {model_name} v{version_info.version} → stage '{target_stage}'"
    )
    return version_info


def load_registered_spell_model(
    model_name: str = REGISTERED_MODEL_NAME,
    *,
    stage: str = "Staging",
):
    """Загружает зарегистрированную модель по stage (None → Staging → Production)."""
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.sklearn.load_model(model_uri)
