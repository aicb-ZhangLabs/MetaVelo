import os

from dataclasses import dataclass, field
from src.slurm import SlurmConfig, slurm_launcher
from src.train.deep_velo import DeepveloConfig, pretrain_deepvelo_model


@dataclass
class ExperimentConfig:
    # model configurations
    model: DeepveloConfig = field(default_factory=DeepveloConfig)

    # slurm configurations
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


@slurm_launcher(ExperimentConfig)
def main(config: ExperimentConfig):
    # create output folder
    os.makedirs(config.model.data_folder, exist_ok=True)
    os.makedirs(config.model.model_folder, exist_ok=True)

    pretrain_deepvelo_model(config.model)


if __name__ == "__main__":
    main()
