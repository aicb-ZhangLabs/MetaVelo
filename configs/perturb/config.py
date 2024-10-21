import tyro

from dataclasses import dataclass, field
from typing import Dict
from src.model.ode import RNAVeloNetConfig
from src.train.cell_type_cls import (
    CellTypeClassifierTrainerConfig,
)
from src.trainer.metavelo import PerturbationTrainerConfig
from src.wandb import WandbConfig
from src.slurm import SlurmConfig
from src.pert import PertModuleConfig

from configs.perturb import pancreas, dentategyrus, bonemarrow


@dataclass
class ExperimentArgs:
    # trainer checkpoint path
    trainer_ckpt_path: str

    # trainer should be resumed
    resume_trainer: bool = False

    # inverse head loss
    inverse_loss: bool = False

    # model configurations
    model: RNAVeloNetConfig = field(default_factory=RNAVeloNetConfig)

    # pert model configurations
    pert: PertModuleConfig = field(default_factory=PertModuleConfig)

    # head model configurations
    head_trainer: CellTypeClassifierTrainerConfig = field(
        default_factory=CellTypeClassifierTrainerConfig
    )

    # trainer arguments
    trainer: PerturbationTrainerConfig = field(
        default_factory=PerturbationTrainerConfig
    )

    # wandb
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # slurm
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


experiments: Dict[str, ExperimentArgs] = {}


experiments["pancreas-default"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    model=pancreas.model,
    pert=PertModuleConfig(),
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-subset-sampling"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    model=pancreas.model,
    pert=pancreas.subset_sampling_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_subset_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-subset-sampling-sampler-subset"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    model=pancreas.model,
    pert=pancreas.subset_sampling_subset_sampler_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_subset_pancreas_sampler_subset",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-subset-sampling-sampler-subset-binary-head"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=pancreas.model,
    pert=pancreas.subset_sampling_subset_sampler_pert,
    head_trainer=pancreas.binary_head_trainer_inverse,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_subset_pancreas_sampler_subset_binary_head",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-sage"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    model=pancreas.model,
    pert=pancreas.sage_pert,
    head_trainer=pancreas.head_trainer_inverse,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_sage_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-sage-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=pancreas.model,
    pert=pancreas.sage_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_sage_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-sage-binary-head"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    model=pancreas.model,
    pert=pancreas.sage_pert,
    head_trainer=pancreas.binary_head_trainer_inverse,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_sage_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-lime-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=pancreas.model,
    pert=pancreas.lime_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_lime_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-cxplain-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=pancreas.model,
    pert=pancreas.cxplain_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_cxplain_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-fimap"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=False,
    model=pancreas.model,
    pert=pancreas.fimap_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_fimap_pancreas",
        notes=pancreas.output_path,
    ),
)

# for some reasons, the permutation here is actually the permutation with feature perturbation not the permutation test
# use `permutation-test` to do the permutation test instead
experiments["pancreas-permutation-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=pancreas.model,
    pert=pancreas.permutation_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_permutation_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-permutation-test"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    model=pancreas.model,
    pert=pancreas.permutation_test_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_permutation_test_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-mean-importance"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    model=pancreas.model,
    pert=pancreas.mean_importance_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_mean_importance_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-feature-ablation"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    model=pancreas.model,
    pert=pancreas.feature_ablation_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_feature_ablation_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-saliency-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=pancreas.model,
    pert=pancreas.saliency_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_saliency_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-smooth-saliency-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=pancreas.model,
    pert=pancreas.smooth_saliency_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_smooth_saliency_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["pancreas-approximation-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=pancreas.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=pancreas.model,
    pert=pancreas.approximation_pert,
    head_trainer=pancreas.head_trainer,
    trainer=pancreas.perturb_trainer,
    slurm=pancreas.slurm,
    wandb=WandbConfig(
        name="pert_approximation_pancreas",
        notes=pancreas.output_path,
    ),
)

experiments["dentategyrus-default"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    model=dentategyrus.model,
    pert=PertModuleConfig(),
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-subset-sampling"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    model=dentategyrus.model,
    pert=dentategyrus.subset_sampling_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_subset_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-subset-sampling-sampler-subset"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    model=dentategyrus.model,
    pert=dentategyrus.subset_sampling_subset_sampler_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_subset_dentategyrus_sampler_subset",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-subset-sampling-sampler-subset-binary-head"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=dentategyrus.model,
    pert=dentategyrus.subset_sampling_subset_sampler_pert,
    head_trainer=dentategyrus.binary_head_trainer_inverse,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_subset_dentategyrus_sampler_subset_binary_head",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-sage"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    model=dentategyrus.model,
    pert=dentategyrus.sage_pert,
    head_trainer=dentategyrus.head_trainer_inverse,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_sage_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-sage-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=dentategyrus.model,
    pert=dentategyrus.sage_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_sage_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-sage-binary-head"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    model=dentategyrus.model,
    pert=dentategyrus.sage_pert,
    head_trainer=dentategyrus.binary_head_trainer_inverse,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_sage_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-lime-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=dentategyrus.model,
    pert=dentategyrus.lime_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_lime_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-cxplain-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=dentategyrus.model,
    pert=dentategyrus.cxplain_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_cxplain_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-fimap"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=False,
    model=dentategyrus.model,
    pert=dentategyrus.fimap_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_fimap_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-permutation-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=dentategyrus.model,
    pert=dentategyrus.permutation_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_permutation_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-permutation-test"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    model=dentategyrus.model,
    pert=dentategyrus.permutation_test_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_permutation_test_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-mean-importance"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    model=dentategyrus.model,
    pert=dentategyrus.mean_importance_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_mean_importance_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-feature-ablation"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    model=dentategyrus.model,
    pert=dentategyrus.feature_ablation_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_feature_ablation_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-saliency-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=dentategyrus.model,
    pert=dentategyrus.saliency_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_saliency_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-smooth-saliency-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=dentategyrus.model,
    pert=dentategyrus.smooth_saliency_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_smooth_saliency_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["dentategyrus-approximation-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=dentategyrus.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=dentategyrus.model,
    pert=dentategyrus.approximation_pert,
    head_trainer=dentategyrus.head_trainer,
    trainer=dentategyrus.perturb_trainer,
    slurm=dentategyrus.slurm,
    wandb=WandbConfig(
        name="pert_approximation_dentategyrus",
        notes=dentategyrus.output_path,
    ),
)

experiments["bonemarrow-default"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    model=bonemarrow.model,
    pert=PertModuleConfig(),
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-subset-sampling"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    model=bonemarrow.model,
    pert=bonemarrow.subset_sampling_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_subset_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-subset-sampling-sampler-subset"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    model=bonemarrow.model,
    pert=bonemarrow.subset_sampling_subset_sampler_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_subset_bonemarrow_sampler_subset",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-subset-sampling-sampler-subset-binary-head"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=bonemarrow.model,
    pert=bonemarrow.subset_sampling_subset_sampler_pert,
    head_trainer=bonemarrow.binary_head_trainer_inverse,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_subset_bonemarrow_sampler_subset_binary_head",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-sage"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    model=bonemarrow.model,
    pert=bonemarrow.sage_pert,
    head_trainer=bonemarrow.head_trainer_inverse,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_sage_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-sage-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=bonemarrow.model,
    pert=bonemarrow.sage_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_sage_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-sage-binary-head"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    model=bonemarrow.model,
    pert=bonemarrow.sage_pert,
    head_trainer=bonemarrow.binary_head_trainer_inverse,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_sage_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-lime-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=bonemarrow.model,
    pert=bonemarrow.lime_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_lime_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-cxplain-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=bonemarrow.model,
    pert=bonemarrow.cxplain_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_cxplain_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-fimap"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=False,
    model=bonemarrow.model,
    pert=bonemarrow.fimap_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_fimap_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-permutation-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=bonemarrow.model,
    pert=bonemarrow.permutation_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_permutation_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-permutation-test"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    model=bonemarrow.model,
    pert=bonemarrow.permutation_test_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_permutation_test_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-mean-importance"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    model=bonemarrow.model,
    pert=bonemarrow.mean_importance_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_mean_importance_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-feature-ablation"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    model=bonemarrow.model,
    pert=bonemarrow.feature_ablation_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_feature_ablation_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-saliency-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=bonemarrow.model,
    pert=bonemarrow.saliency_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_saliency_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-smooth-saliency-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=bonemarrow.model,
    pert=bonemarrow.smooth_saliency_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_smooth_saliency_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

experiments["bonemarrow-approximation-inverse-loss"] = ExperimentArgs(
    trainer_ckpt_path=bonemarrow.trainer_ckpt_path,
    resume_trainer=False,
    inverse_loss=True,
    model=bonemarrow.model,
    pert=bonemarrow.approximation_pert,
    head_trainer=bonemarrow.head_trainer,
    trainer=bonemarrow.perturb_trainer,
    slurm=bonemarrow.slurm,
    wandb=WandbConfig(
        name="pert_approximation_bonemarrow",
        notes=bonemarrow.output_path,
    ),
)

ConfiguredExperimentArgs: type[ExperimentArgs] = (
    tyro.extras.subcommand_type_from_defaults(experiments)
)
