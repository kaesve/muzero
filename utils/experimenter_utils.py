from itertools import product
from os import listdir
import subprocess as sp
import typing

import Coach
from utils import DotDict


def create_parameter_grid(content: DotDict) -> typing.List:
    """
    Recursively build up a parameter-grid using itertools.product on all dict values within content
    :param content: DotDict dictionary with keys accessible as object attributes
    :return: List of DotDict objects containing a parameter grid for each value given per key.
    :reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html
    """
    # Recursively unpack dictionary to create flat grids at each level.
    base = DotDict()
    for key, value in content.items():
        if isinstance(value, DotDict):
            base[key] = create_parameter_grid(value)
        else:
            base[key] = value

    # Build up a list of dictionaries for each possible value combination.
    grid = list()
    keys, values = zip(*base.items())
    for v in product(*values):
        grid.append(DotDict(zip(keys, v)))

    return grid


def get_gpu_memory():
    cmd = "nvidia-smi --query-gpu=memory.free --format=csv"
    output = sp.check_output(cmd.split())
    memory_free_info = (output.decode('ascii').split('\n')[:-1])[1:]

    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

    return memory_free_values


def get_player_pool(player_configs: typing.List, by_checkpoint: bool = False, resolution: int = 1) -> typing.List:
    """
    Build up a player pool of the form (Player-Class, details) where details is either None for non parametric implementations
    or an unpacked tuple of a model's configuration path and filename (i.e., where to find its data/ weights).

    If an agent is non-parametric, its player representation will look like: (PlayerClass, None)
    If an agent is parametric, its player representation will look like: (PlayerClass, folder_path, filename)

    Each object in the player_configs is an interface that represents the base agent, the data contained within
    this class must contain a boolean whether it requires to load in data/ weights and where to find these weights.
    If by_checkpoint is specified, we will define every x (resolution) checkpoints of the model as an individual
    player, otherwise we only use the latest one (specified in the player args).

    We do not make copy's of the Agent classes/ interface to cleverly reuse objects to save memory requirements
    when using large implementations.

    :param player_configs: List of base player configs. This represents the base agent interface/ data structure.
    :param by_checkpoint: bool Whether to load in all available checkpoints as players.
    :param resolution: int Resolution to load in every x checkpoint of the by_checkpoint option.
    :return: List of players of either tuples (non-parametric) or triples (parametric).
    """
    player_pool = list()
    for p in player_configs:
        if p.parametric:
            if by_checkpoint:  # Load in each ('resolution') checkpoint of the model.
                path, _ = p.args.args.load_folder_file

                # Get all checkpoint files in folder by int, and create new accessing filenames.
                model_files = [s for s in listdir(path) if s.endswith('.pth.tar') and 'checkpoint' in s]
                checkpoint_ints = {int(s.split('_')[-1].split('.')[0]) for s in model_files}
                checkpoints = [(p, path, Coach.Coach.getCheckpointFile(int(s))) for s in checkpoint_ints
                               if not int(s) % resolution]

                player_pool += checkpoints
            else:  # Load in best/ final model.
                player_pool.append((p, *p.args.args.load_folder_file))
        else:
            player_pool.append((p, None))

    return player_pool
