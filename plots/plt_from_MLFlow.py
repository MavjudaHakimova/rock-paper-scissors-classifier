import os
import mlflow
from pathlib import Path

def save_metrics_plots(trainer, experiment_id=None, run_id=None):
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    if run_id is None:
        # Получаем последний запуск текущего эксперимента
        client = mlflow.tracking.MlflowClient()
        if experiment_id is None:
            experiment = client.get_experiment_by_name(trainer.logger.experiment_name)
            experiment_id = experiment.experiment_id
        runs = client.search_runs(experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            run_id = runs[0].info.run_id
        else:
            print("Не найдено ни одного запуска в MLflow!")
            return

    # Получаем список артефактов (графиков)
    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id, path="plots/")
    for artifact in artifacts:
        if artifact.path.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
            # Загружаем артефакт
            local_path = plots_dir / artifact.path.split("/")[-1]
            mlflow.artifacts.download_artifacts(run_id=run_id, path=artifact.path, dst_path=str(local_path))
            print(f"График сохранён: {local_path}")
