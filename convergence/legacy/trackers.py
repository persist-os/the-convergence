"""
Optional external tracker implementations.

These are placeholders for future implementation.
Current version uses builtin tracking (SQLite + CSV).

Future trackers:
- MLflow: Free, self-hosted experiment tracking
- Aim: Lightweight, modern UI
- Weave: W&B hosted (paid)
"""

# TODO: Implement in future release
# 
# class MLflowTracker:
#     """MLflow integration (free, self-hosted)."""
#     
#     def __init__(self, config: dict):
#         import mlflow
#         self.mlflow = mlflow
#         mlflow.set_tracking_uri(config.get("tracking_uri", "sqlite:///data/mlruns.db"))
#         mlflow.set_experiment(config.get("experiment_name", "convergence"))
#     
#     async def log_run(self, run: OptimizationRun):
#         """Log run to MLflow."""
#         with self.mlflow.start_run(run_name=run.run_id):
#             self.mlflow.log_params(run.config)
#             self.mlflow.log_metrics(run.aggregate_metrics)
#             self.mlflow.log_metric("aggregate_score", run.aggregate_score)
#             self.mlflow.log_metric("duration_ms", run.duration_ms)
#             self.mlflow.log_metric("cost_usd", run.cost_usd)
#
#
# class AimTracker:
#     """Aim integration (free, lightweight)."""
#     
#     def __init__(self, config: dict):
#         from aim import Run
#         self.Run = Run
#         self.repo = config.get("repo", "./data/aim_repo")
#         self.run = None
#     
#     async def log_run(self, run: OptimizationRun):
#         """Log run to Aim."""
#         aim_run = self.Run(repo=self.repo, experiment=run.session_id)
#         aim_run.track(run.aggregate_metrics, context={"config": run.config})
#         aim_run.track(run.aggregate_score, name="score")
#
#
# class WeaveTracker:
#     """Weave integration (optional, paid)."""
#     
#     def __init__(self, config: dict):
#         import weave
#         self.weave = weave
#         weave.init(config.get("project"))
#     
#     async def log_run(self, run: OptimizationRun):
#         """Log run to Weave."""
#         self.weave.log({"run": run.dict()})


# Note: To enable these trackers in the future:
# 1. Install the package: pip install mlflow / pip install aim / etc.
# 2. Uncomment the relevant class above
# 3. Update LegacyStore._init_tracker() to instantiate the tracker
# 4. Add configuration to YAML under legacy.tracking section

