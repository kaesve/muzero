from .Arena import Arena
from .experimenter import ExperimentConfig, tournament_pool, tournament_final, run_ablations

experiments = {
    'TOURNEY': tournament_final,
    'CHECKPOINT_TOURNEY': tournament_pool,
    'TRAIN_GRID': run_ablations
}
