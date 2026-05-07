"""
ClearML experiment logging (replaces Neptune).

Credentials: set via one of:
  - Interactive: clearml-init  (creates ~/clearml.conf)
  - Env vars: CLEARML_API_HOST, CLEARML_WEB_HOST, CLEARML_API_KEY, CLEARML_API_SECRET
  - Or: CLEARML_CONFIG_FILE=/path/to/clearml.conf

See: https://clear.ml/docs/latest/docs/configs/configuration/
"""

import yaml
import importlib
import numpy as np
from copy import deepcopy
from typing import Callable, Any, Dict

# Optional: only import when ClearML is used
try:
    from clearml import Task
    from clearml import Logger
    _CLEARML_AVAILABLE = True
except ImportError:
    _CLEARML_AVAILABLE = False
    Task = None
    Logger = None

import optuna


def _stringify_unsupported(obj: Any) -> Any:
    """Convert non-JSON-serializable values to strings (like Neptune's stringify_unsupported)."""
    if isinstance(obj, dict):
        return {k: _stringify_unsupported(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify_unsupported(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


class ClearmlRun:
    """
    Wrapper that mirrors Neptune Run interface for logging:
      run['key'] = value
      run['key'].append(value)
      run['path'].upload(matplotlib_figure)
    """

    def __init__(self, task: "Task"):
        if not _CLEARML_AVAILABLE:
            raise RuntimeError("ClearML is not installed. Install with: uv add clearml")
        self._task = task
        self._logger = task.get_logger()
        self._iteration = 0
        self._batch_step = 0

    def set_iteration(self, epoch: int):
        """Set current epoch for epoch-level scalar logging."""
        self._iteration = epoch

    def __getitem__(self, key: str):
        return _ClearmlChannel(self, key)

    def __setitem__(self, key: str, value: Any):
        if key in ("config.yaml", "params", "sys/group_tags", "sys/id"):
            return
        if isinstance(value, (int, float, np.integer, np.floating)):
            self._logger.report_single_value(key.replace("/", "_"), float(value))
        elif isinstance(value, str):
            self._logger.report_text(value, level=20)
        else:
            self._logger.report_single_value(key.replace("/", "_"), str(value))

    def _report_scalar(self, key: str, value: float, use_batch_step: bool = False):
        parts = key.replace("train/", "").replace("trial/", "").split("/")
        title = parts[0] if len(parts) > 0 else "scalar"
        series = "/".join(parts[1:]) if len(parts) > 1 else key.replace("/", "_")
        it = self._batch_step if use_batch_step else self._iteration
        self._logger.report_scalar(title=title, series=series, value=float(value), iteration=it)
        if use_batch_step:
            self._batch_step += 1

    def _report_dict_scalars(self, key: str, d: Dict[str, float]):
        """Log each key in d as a scalar at current iteration (title=key, series=subkey)."""
        title = key.split("/")[0] or "metrics"
        for k, v in d.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                self._logger.report_scalar(
                    title=title, series=k, value=float(v), iteration=self._iteration
                )
            elif isinstance(v, dict):
                self._report_dict_scalars(f"{key}/{k}", v)

    def _report_figure(self, key: str, fig):
        parts = key.replace("train/", "").split("/")
        title = parts[0] if parts else "image"
        series = "/".join(parts[1:]) if len(parts) > 1 else key.replace("/", "_")
        self._logger.report_matplotlib_figure(
            title=title, series=series, figure=fig, iteration=self._iteration
        )

    def get_task_id(self) -> str:
        return self._task.id


class DummyRun:
    """
    No-op run for inference or when logging is disabled.
    Same interface as ClearmlRun: run['key'].append(), run['key'].upload(), run.get_task_id().
    """

    def __init__(self, task_id: str = "inference"):
        self._task_id = task_id

    def set_iteration(self, epoch: int):
        pass

    def __getitem__(self, key: str):
        return _DummyChannel()

    def __setitem__(self, key: str, value: Any):
        pass

    def get_task_id(self) -> str:
        return self._task_id


class _DummyChannel:
    def append(self, value):
        pass

    def upload(self, fig):
        pass

    def add(self, values):
        pass


class _ClearmlChannel:
    """Adapter so that run['key'].append(value) and run['key'].upload(fig) work."""

    def __init__(self, run: ClearmlRun, key: str):
        self._run = run
        self._key = key

    def append(self, value):
        if isinstance(value, dict):
            self._run._report_dict_scalars(self._key, value)
        else:
            use_batch = "batch" in self._key
            self._run._report_scalar(self._key, float(value), use_batch_step=use_batch)

    def upload(self, fig):
        self._run._report_figure(self._key, fig)

    def add(self, values):
        """Neptune's run['sys/group_tags'].add([...]). No-op for ClearML (tags can be set on Task)."""
        pass


def parse_sampling_strategy(sampling_strategy: str, *args, **kwargs):
    """Load Optuna sampler by name."""
    available = [
        "RandomSampler", "GridSampler", "TPESampler", "CmaEsSampler",
        "NSGAIISampler", "QMCSampler", "GPSampler", "BoTorchSampler", "BruteForceSampler",
    ]
    if sampling_strategy not in available:
        print("Sampler not specified correctly, use default TPESampler")
        return None
    module = importlib.import_module("optuna.samplers")
    sampler_class = getattr(module, sampling_strategy)
    return sampler_class(*args, **kwargs)


def if_study(config: dict) -> bool:
    """True if config describes an Optuna study (has 'distribution' in nested dicts)."""
    for value in config.values():
        if isinstance(value, dict):
            if "distribution" in value:
                return True
            if if_study(value):
                return True
    return False


def get_trial_config(config: dict, trial: optuna.Trial) -> dict:
    """Sample config from trial distributions."""
    trial_config = {}
    for key, value in deepcopy(config).items():
        if isinstance(value, dict):
            if "distribution" in value:
                _suggest_variable(trial_config, key, value, trial)
            else:
                value.update(get_trial_config(value, trial))
                trial_config[key] = value
        else:
            trial_config[key] = value
    return trial_config


def _suggest_variable(trial_config: dict, key: str, value: dict, trial: optuna.Trial):
    d = value["distribution"]
    if d == "categorical":
        trial_config[key] = trial.suggest_categorical(key, value["values"])
    elif d == "discrete_uniform":
        trial_config[key] = trial.suggest_discrete_uniform(
            key, value["low"], value["high"], value["q"]
        )
    elif d == "float":
        trial_config[key] = trial.suggest_float(
            key, value["low"], value["high"], step=value["step"], log=value["log"]
        )
    elif d == "int":
        step = 1 if value["log"] else value["step"]
        trial_config[key] = trial.suggest_int(
            key, value["low"], value["high"], step=step, log=value["log"]
        )
    else:
        raise ValueError(f"Unknown distribution: {d}")


class ClearMLManager:
    """
    Manages ClearML tasks and Optuna studies (replaces NeptuneManager).
    """

    def __init__(self, project: str, config: dict, name: str = ""):
        if not _CLEARML_AVAILABLE:
            raise RuntimeError("ClearML is not installed. Install with: uv add clearml")
        self.project = project
        self.config = config
        self.name = name or "gen3d_run"
        self.sweep = if_study(config)
        self.study = None
        self.run = None
        self._task = None

        self._task = Task.init(
            project_name=project,
            task_name=name or "gen3d_run",
            tags=["study" if self.sweep else "trial"],
        )
        self._task.connect(config, name="config")
        self.run = ClearmlRun(self._task)
        self._task.get_logger().report_text(
            "Config (YAML):\n" + yaml.dump(config, sort_keys=False)
        )
        self._task.set_parameter("params", _stringify_unsupported(config))

        if self.sweep:
            self.metrics = config["metrics"]
            self.n_metrics = len(self.metrics)
            self.directions = [m["direction"] for m in self.metrics]
            self.names = [m["name"] for m in self.metrics]
            self.if_logs = [
                m.get("log", False) for m in self.metrics
            ]

    def optimize(
        self,
        inner_objective: Callable[[dict, ClearmlRun], dict],
        sampler: str = "TPESampler",
        n_trials: int = 1000,
        storage: str | None = None,
        study_name: str | None = None,
    ):
        """Run Optuna optimization; each trial gets its own ClearML task."""
        sampler_obj = parse_sampling_strategy(sampler)
        resolved_study_name = study_name or self.name
        self.study = optuna.create_study(
            directions=self.directions,
            sampler=sampler_obj,
            study_name=resolved_study_name,
            storage=storage,
            load_if_exists=True,
        )
        self.study.set_metric_names(self.names)

        def objective(trial: optuna.Trial):
            trial_config = get_trial_config(self.config, trial)
            task = Task.init(
                project_name=self.project,
                task_name=f"{self.name}_trial_{trial.number}",
                tags=["trial", self.study.study_name],
            )
            task.connect(trial_config, name="config")
            run_trial = ClearmlRun(task)
            run_trial["trial/number"] = trial.number
            run_trial.set_iteration(0)
            score = inner_objective(trial_config, run_trial)
            metrics = []
            for if_log, metric_name in zip(self.if_logs, self.names):
                val = score[metric_name]
                if if_log:
                    val = np.log(val)
                run_trial[f"trial/{metric_name}"] = val
                metrics.append(val)
            task.close()
            return metrics

        self.study.optimize(objective, n_trials=n_trials)

    def finish(self):
        """Close the main task."""
        if self._task:
            self._task.close()
