from __future__ import annotations
import os
import typing
import time
from concurrent.futures.thread import ThreadPoolExecutor
import subprocess as sp
from datetime import datetime

import numpy as np

from utils import DotDict
from utils.experimenter_utils import get_gpu_memory
from .experimenter import ExperimentConfig


class AblationAnalysis:

    def __init__(self, experiment: ExperimentConfig, config_dir: str = './temp/') -> None:
        self.experiment = experiment
        self.config_dir = config_dir

        self.configs = list()
        self.files = list()

    def __enter__(self) -> AblationAnalysis:
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

        self.configs = list()
        base_config = DotDict.from_json(self.experiment.ablation_base.config)
        for param in self.experiment.ablation_grid:
            config = base_config.copy()
            config.recursive_update(param)
            self.configs.append(config)

        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        for run in range(self.experiment.experiment_args.num_repeat):
            for i, config in enumerate(self.configs):
                c = config.copy()  # Note: shallow copy.

                run_config_name = f'rep{run}_config{i}_dt{dt}'
                c.name = f'{c.name}_{run_config_name}'

                out = os.path.join(self.experiment.output_directory, c.args.checkpoint, run_config_name)
                c.args.checkpoint = out
                c.args.load_folder_file = (out, c.args.load_folder_file[1])

                if not os.path.exists(c.args.checkpoint):
                    os.makedirs(c.args.checkpoint)

                config_file = os.path.join(self.config_dir, run_config_name) + '.json'
                c.to_json(config_file)

                self.files.append(config_file)

        return self

    def run(self):
        execute = True

        def start_run(config: str, flags: str) -> None:
            if execute:
                gpu_memory = get_gpu_memory()
                gpu = np.argmax(gpu_memory)
                print(f"Best GPU to use: {gpu} with {gpu_memory[gpu]} MiB available VRAM.")

                cmd = f'python Main.py train -c {config} {flags} '
                if '--gpu' not in cmd:  # If CUDA should not be used (CPU) --> set '--cpu -1' in config flags.
                    cmd += f'--gpu {gpu}'

                sp.call(cmd.split())

        num_threads = self.experiment.experiment_args.n_jobs
        print(f'Constructing ThreadPool. Queue size: {len(self.files)} jobs, using {num_threads} threads')

        try:
            tp = ThreadPoolExecutor(max_workers=num_threads)
            job = list()
            for file in self.files:
                time.sleep(1)  # Give threads some time to start up and select GPU.
                job.append(tp.submit(start_run, file, self.experiment.experiment_args.flags))

            while not all(j.done() for j in job):
                time.sleep(1)

        except KeyboardInterrupt:
            print("Keyboard Interrupt. Giving workers time to terminate.")
            execute = False

            time.sleep(1)
            print("Workers have exited.")

        print('All processes have exited.')

    def __exit__(self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any) -> None:
        # Remove every used temporary file.
        for file in self.files:
            os.remove(file)

        # Remove dir if not used for other purposes
        if not os.listdir(self.config_dir):
            os.rmdir(self.config_dir)


def run_ablations(experiment: ExperimentConfig) -> None:

    with AblationAnalysis(experiment) as ab:

        ab.run()

