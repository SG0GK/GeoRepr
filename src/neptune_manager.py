import yaml
import importlib
import neptune
import optuna
import neptune.integrations.optuna as npt_utils
from neptune.utils import stringify_unsupported
from neptune.types import GitRef
from copy import deepcopy
import abc
from typing import Callable


def parse_sampling_strategy(sampling_strategy: str, *args, **kwargs) -> abc.ABCMeta:
    """Loads optuna sampler

    Args:
        sampling_strategy (str): Name of optuna sampler.

    Returns:
        abc.ABCMeta: Sampler module from optuna
    """
    available_strat = ['RandomSampler', 'GridSampler', 'TPESampler',
                       'CmaEsSampler', 'NSGAIISampler', 'QMCSampler',
                       'GPSampler', 'BoTorchSampler', 'BruteForceSampler']
    if sampling_strategy not in available_strat:
        print('Sampler is not specified correctly, use default TPESampler')
        return None
    else:
        module_name = 'optuna.samplers'
        module = importlib.import_module(module_name)
        sampler_class = getattr(module, sampling_strategy)
        sampler = sampler_class(*args, **kwargs)
        print('Sampler imported successfully!')
        return sampler


def if_study(config: dict) -> bool:
    """Checks if config represents an optuna study

    Args:
        config (dict): dictionary with model training configuration

    Returns:
        bool: True if it is study, False if it is single run
    """
    tag = False
    for value in config.values():
        if isinstance(value, dict):
            if 'distribution' in value.keys():
                tag = True
            else:
                tag = if_study(value) | tag
    return tag


def get_trial_config(config: dict, trial: optuna.trial.Trial) -> dict:
    """Detects which parameters in config we need to optimized.
    Detection happens when some parameter in config has "distribution" in its keys.
    Then it samples those parameters from their distribuions.
    In the end it returns config with sampled data.

    Args:
        config (dict): study config
        trial (optuna.trial.Trial): current optuna trial (study)

    Returns:
        dict: trial config with sampled parameters
    """
    trial_config = dict()
    for key, value in deepcopy(config).items():
        if isinstance(value, dict):
            if 'distribution' in value.keys():
                suggest_variable(trial_config, key, value, trial)
            else:
                value.update(get_trial_config(value, trial))
                trial_config[key] = value
        else:
            trial_config[key] = value
    return trial_config


def suggest_variable(trial_config: dict, key: str, value: dict, trial: optuna.trial.Trial):
    """Sampling parameters for optimization

    Args:
        trial_config (dict): config with parameters
        key (str): name of parameter
        value (dict): parameter description, must include distribution
        trial (optuna.trial.Trial): current optuna trial (study)

    Raises:
        ValueError: when the distribution is unkown
    """
    if value['distribution'] == 'categorical':
        trial_config[key] = trial.suggest_categorical(
            name=key, choices=value['values']
        )
    elif value['distribution'] == 'discrete_uniform':
        trial_config[key] = trial.suggest_discrete_uniform(
            name=key, low=value['low'], high=value['high'], q=value['q']
        )
    elif value['distribution'] == 'float':
        trial_config[key] = trial.suggest_float(
            name=key, low=value['low'], high=value['high'], step=value['step'], log=value['log']
        )
    elif value['distribution'] == 'int':
        step = 1 if value['log'] else value['step']
        trial_config[key] = trial.suggest_int(
            name=key, low=value['low'], high=value['high'], step=step, log=value['log']
        )
    else:
        raise ValueError(f"Uknown variable distribution: {value['distribution']}")


class NeptuneManager(object):

    def __init__(self, project: str, config: dict, name: str = '') -> None:
        """Class for managing runs and studies of runs.
        Starts a main neptune run for the whole code.
        If the config descirbes a study (sweep), the main run is a study run.
        If the config describes a single run, the main run is a trial run.

        Args:
            project (str): Name of neptune project
            config (dict): Config of study or single run
            name (str, optional): name of run or study chosen by user. Defaults to ''.
        """
        self.project, self.config, self.name = project, config, name
        tag, self.sweep, self.study = 'trial', False, None
        if if_study(config):
            tag, self.sweep = 'study', True
            self.metrics, self.n_metrics = config['metrics'], len(config['metrics'])
            self.directions = list([metric['direction'] for metric in self.metrics])
            self.names = list([metric['name'] for metric in self.metrics])
        self.run = neptune.init_run(project=project, name=name, git_ref=GitRef.DISABLED,
                                    source_files=[], tags=[tag])
        self.run['config.yaml'] = yaml.dump(config, sort_keys=False)
        self.run['params'] = stringify_unsupported(config)
        self.neptune_callback = npt_utils.NeptuneCallback(self.run)

    def optimize(self, inner_objective: Callable[[dict, neptune.Run], dict], sampler='TPESampler', n_trials=100):
        """Run optimization of the function called inner_objective

        Args:
            inner_objective (Callable[[dict, neptune.Run], dict]): function that
                takes parameters config and local neptune run and returns dictionary with metrics
            sampler (str, optional): sampler name. Defaults to 'TPESampler'.
            n_trials (int, optional): number of trials. Defaults to 100.
        """
        self.sampler(inner_objective)
        sampler = parse_sampling_strategy(sampler)
        self.study = optuna.create_study(directions=self.directions, sampler=sampler,
                                         study_name=self.name)
        self.study.set_metric_names(self.names)
        self.run["sys/group_tags"].add([self.study.study_name])
        self.study.optimize(self.objective, n_trials=n_trials, callbacks=[self.neptune_callback])
        npt_utils.log_study_metadata(self.study, self.run)

    def sampler(self, inner_objective: Callable[[dict, neptune.Run], dict]):
        """Wraps up the inner objective into its separate neptune run.

        Args:
            inner_objective (Callable[[dict, neptune.Run], dict]): function that
                takes parameters config and local neptune run and returns dictionary with metrics
        """
        def objective(trial: optuna.trial.Trial):
            trial_config = get_trial_config(self.config, trial)
            run_trial_level = neptune.init_run(project=self.project, git_ref=GitRef.DISABLED,
                                               tags=["trial"], source_files=[])
            run_trial_level["config.yaml"] = yaml.dump(trial_config, sort_keys=False)
            run_trial_level["sys/group_tags"].add([self.study.study_name])
            run_trial_level["trial/number"] = trial.number
            run_trial_level["params"] = stringify_unsupported(trial_config)
            score = inner_objective(trial_config, run_trial_level)
            metrics = list()
            for metric in self.names:
                run_trial_level[f"trial/{metric}"] = score[metric]
                metrics.append(score[metric])
            run_trial_level.stop()
            return metrics
        self.objective = objective

    def finish(self):
        """Stopping the main run
        """
        self.run.stop()
