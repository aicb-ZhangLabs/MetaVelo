import tyro
from typing import Literal, Dict
from dataclasses import dataclass, field
from src.train.synthetic_data import (
    SyntheticDataClassifierTrainerConfig,
)
from src.trainer.synthetic_data import (
    SyntheticDataPerturbationTrainerConfig,
)
from src.wandb import WandbConfig
from src.slurm import SlurmConfig
from src.pert import PertModuleConfig
from configs.perturb_synthetic_data import synthetic_data


@dataclass
class ExperimentArgs:
    # trainer should be resumed
    resume_trainer: bool = False

    # inverse head loss
    inverse_loss: bool = False

    # perturbation type
    perturbation_value_type: Literal["zero", "one"] = "zero"

    # pert model configurations
    pert: PertModuleConfig = field(default_factory=PertModuleConfig)

    # head model configurations
    head_trainer: SyntheticDataClassifierTrainerConfig = field(
        default_factory=SyntheticDataClassifierTrainerConfig
    )

    # trainer arguments
    trainer: SyntheticDataPerturbationTrainerConfig = field(
        default_factory=SyntheticDataPerturbationTrainerConfig
    )

    # wandb
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # slurm
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


experiments: Dict[str, ExperimentArgs] = {}
experiments["nonlinear-additive-default"] = ExperimentArgs(
    resume_trainer=False,
    perturbation_value_type="-one",
    pert=synthetic_data.subset_sampling_subset_sampler_pert,
    head_trainer=synthetic_data.nonlinear_additive_head_trainer,
    trainer=synthetic_data.nonlinear_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_subset_synthetic_data_sampler_subset",
        notes=synthetic_data.output_path,
    ),
)

experiments["nonlinear-additive-subset-sampling-sampler-subset"] = ExperimentArgs(
    resume_trainer=False,
    perturbation_value_type="-one",
    pert=synthetic_data.subset_sampling_subset_sampler_pert,
    head_trainer=synthetic_data.nonlinear_additive_head_trainer,
    trainer=synthetic_data.nonlinear_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_subset_synthetic_data_sampler_subset",
        notes=synthetic_data.output_path,
    ),
)

experiments["orange-skin-additive-subset-sampling-sampler-subset"] = ExperimentArgs(
    resume_trainer=False,
    perturbation_value_type="-one",
    pert=synthetic_data.subset_sampling_subset_sampler_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_subset_synthetic_data_sampler_subset",
        notes=synthetic_data.output_path,
    ),
)

experiments["nonlinear-additive-sage-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.sage_pert,
    head_trainer=synthetic_data.nonlinear_additive_head_trainer,
    trainer=synthetic_data.nonlinear_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_sage_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)

experiments["orange-skin-additive-sage-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.sage_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_sage_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)

experiments["nonlinear-additive-approximation-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.approximation_pert,
    head_trainer=synthetic_data.nonlinear_additive_head_trainer,
    trainer=synthetic_data.nonlinear_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_approximation_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)

experiments["orange-skin-additive-approximation-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.approximation_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_approximation_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)

experiments["orange-skin-additive-lime-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.lime_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_lime_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)

experiments["orange-skin-additive-cxplain-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.cxplain_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_cxplain_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)

experiments["orange-skin-additive-fimap-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.fimap_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_fimap_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)

experiments["orange-skin-additive-permutation-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.permutation_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_permutation_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)
experiments["orange-skin-additive-saliency-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.saliency_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_saliency_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)
experiments["orange-skin-additive-smooth_saliency-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.smooth_saliency_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_smooth_saliency_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)
experiments["orange-skin-additive-permutation_test-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.permutation_test_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_permutation_test_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)
experiments["orange-skin-additive-mean_importance-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.mean_importance_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_mean_importance_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)
experiments["orange-skin-additive-feature_ablation-inverse-loss"] = ExperimentArgs(
    resume_trainer=False,
    inverse_loss=True,
    perturbation_value_type="-one",
    pert=synthetic_data.feature_ablation_pert,
    head_trainer=synthetic_data.orange_skin_additive_head_trainer,
    trainer=synthetic_data.orange_skin_additive_perturb_trainer,
    slurm=synthetic_data.slurm,
    wandb=WandbConfig(
        name="pert_feature_ablation_synthetic_data",
        notes=synthetic_data.output_path,
    ),
)

ConfiguredExperimentArgs: type[ExperimentArgs] = (
    tyro.extras.subcommand_type_from_defaults(experiments)
)
