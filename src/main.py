import argparse

from utils import read_yaml
from data import DataCubes
from train.cnn import TrainCNN
from neptune_manager import NeptuneManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-p', '--project', type=str, default='GeoR/Representations')
    parser.add_argument('-s', '--sampler', type=str, default='TPESampler')
    parser.add_argument('-t', '--trials', type=int, default=100)
    parser.add_argument('-n', '--name', type=str, default='')
    parser.add_argument('-m', '--mode', type=str)
    args = parser.parse_args()

    project = args.project
    config = read_yaml(args.config)

    neptune_manager = NeptuneManager(project=project, config=config, name=args.name)

    data = DataCubes(config)
    # X, _, y = data.load_data()
    data.get_dataset(*data.load_data())

    print("Starting neptune")
    if neptune_manager.sweep:
        def inner_objective(trial_config, run_trial_level):
            train = TrainCNN(trial_config, data, run_trial_level)
            best_losses = train.train()
            train.save()
            return best_losses
        neptune_manager.optimize(inner_objective, sampler=args.sampler)
    else:
        train = TrainCNN(config, data, neptune_manager.run)
        train.train()
        train.save()
    neptune_manager.finish()


if __name__ == '__main__':
    main()
