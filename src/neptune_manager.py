import yaml
import importlib
import neptune
import optuna
import neptune.integrations.optuna as npt_utils
from neptune.utils import stringify_unsupported
from neptune.types import GitRef
from copy import deepcopy


def parse_sampling_strategy(sampling_strategy: str, *args, **kwargs):
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


def if_study(config):
    tag = False
    for value in config.values():
        if isinstance(value, dict):
            if 'distribution' in value.keys():
                tag = True
            else:
                tag = if_study(value) | tag
    return tag


def get_trial_config(config, trial: optuna.trial.Trial):
    trial_config = dict()
    for key, value in deepcopy(config).items():
        if isinstance(value, dict):
            if 'distribution' in value.keys():
                if value['distribution'] == 'categorical':
                    trial_config[key] = trial.suggest_categorical(name=key, choices=value['values'])
                if value['distribution'] == 'discrete_uniform':
                    trial_config[key] = trial.suggest_discrete_uniform(name=key, low=value['low'],
                                                                       high=value['high'], q=value['q'])
                if value['distribution'] == 'float':
                    if value.get('log', False):
                        # For log-scale, don't use step parameter
                        trial_config[key] = trial.suggest_float(name=key, low=value['low'],
                                                              high=value['high'], log=True)
                    else:
                        # For linear-scale, use step if provided
                        step = value.get('step', None)
                        trial_config[key] = trial.suggest_float(name=key, low=value['low'],
                                                              high=value['high'],
                                                              step=step, log=False)
                if value['distribution'] == 'int':
                    if value.get('log', False):
                        # For log-scale integers, don't use step
                        trial_config[key] = trial.suggest_int(name=key, low=value['low'],
                                                            high=value['high'], log=True)
                    else:
                        # For linear-scale integers, use step if provided
                        step = value.get('step', 1)
                        trial_config[key] = trial.suggest_int(name=key, low=value['low'],
                                                            high=value['high'],
                                                            step=step, log=False)
            else:
                value.update(get_trial_config(value, trial))
                trial_config[key] = value
        else:
            trial_config[key] = value
    return trial_config


class NeptuneManager(object):

    def __init__(self, project, config, name='') -> None:
        self.project, self.config, self.name = project, config, name
        tag, self.sweep, self.study = 'trial', False, None
        if if_study(config):
            tag, self.sweep = 'study', True
            self.metrics, self.n_metrics = config['metrics'], len(config['metrics'])
            self.directions = list([metric['direction'] for metric in self.metrics])
            self.names = list([metric['name'] for metric in self.metrics])
            
            # For Pareto front visualization, select primary metrics if more than 3 objectives
            if self.n_metrics > 3:
                self.viz_metrics = ['loss', 'view_similarity']  # Primary metrics for SSL
                if 'visualization' in config and 'pareto_metrics' in config['visualization']:
                    self.viz_metrics = config['visualization']['pareto_metrics']
            else:
                self.viz_metrics = self.names
                
        self.run = neptune.init_run(project=project, name=name, git_ref=GitRef.DISABLED,
                                    source_files=[], tags=[tag])
        self.run['config.yaml'] = yaml.dump(config, sort_keys=False)
        self.run['params'] = stringify_unsupported(config)
        self.neptune_callback = npt_utils.NeptuneCallback(self.run)

    def optimize(self, inner_objective, sampler='TPESampler', n_trials=100):
        self.sampler(inner_objective)
        sampler = parse_sampling_strategy(sampler)
        self.study = optuna.create_study(directions=self.directions, sampler=sampler,
                                         study_name=self.name)
        self.study.set_metric_names(self.names)
        self.run["sys/group_tags"].add([self.study.study_name])
        self.study.optimize(self.objective, n_trials=n_trials, callbacks=[self.neptune_callback])
        
        # Log study metadata with proper visualization targets
        if self.n_metrics > 3:
            viz_indices = [self.names.index(metric) for metric in self.viz_metrics]
            npt_utils.log_study_metadata(self.study, self.run, targets=viz_indices)
        else:
            npt_utils.log_study_metadata(self.study, self.run)

    def sampler(self, inner_objective):
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
        self.run.stop()
