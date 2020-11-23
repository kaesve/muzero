from .Arena import Arena
from .experimenter import ExperimentConfig, perform_tournament
from .Parameters import run_ablations

experiments = {
    'TOURNEY': lambda experiment_config: perform_tournament(experiment_config, by_checkpoint=False),
    'CHECKPOINT_TOURNEY': lambda experiment_config: perform_tournament(experiment_config, by_checkpoint=True),
    'TRAIN_GRID': run_ablations
}
