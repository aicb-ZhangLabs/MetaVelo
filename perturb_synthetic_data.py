import torch
import torch.nn as nn

from dataclasses import asdict
from src.model.classifier import DeeperFFNClassifier
from src.train.synthetic_data import (
    SyntheticDataClassifierTrainer,
)
from src.trainer.synthetic_data import (
    SyntheticDataPerturbationTrainer,
    SyntheticDataSAGEPerturbationTrainer,
    SyntheticDataCXPlainPerturbationTrainer,
    SyntheticDataLIMEPerturbationTrainer,
    SyntheticDataFIMAPPerturbationTrainer,
    SyntheticDataPermutationPerturbationTrainer,
    SyntheticDataSaliencyPerturbationTrainer,
)
from src.wandb import init_wandb
from src.slurm import slurm_launcher
from src.pert import PertModule
from src.dispatch import dispatch_pert_module
from src.head.cls import NeuralClassifierHead
from src.utils import seed_everything
from configs.perturb_synthetic_data.config import (
    ConfiguredExperimentArgs,
    ExperimentArgs,
)


@slurm_launcher(ConfiguredExperimentArgs)
def main(args: ExperimentArgs):
    seed_everything(args.trainer.seed)
    init_wandb(args.wandb, asdict(args))

    # model input
    input_dim = args.head_trainer.synthetic_data_feature_dim

    # train head model
    head_layer = DeeperFFNClassifier(
        input_dim,
        len(args.head_trainer.num_type_list),
        hidden_dim=args.head_trainer.hidden_dim,
    )
    head_trainer = SyntheticDataClassifierTrainer(
        head_layer, args=args.head_trainer, has_wandb_writer=True
    )
    head_trainer.train()

    head_model = NeuralClassifierHead(
        head_layer,
        tgt_ind=args.head_trainer.num_type_list.index(args.head_trainer.tgt_num_type),
        inverse_loss=args.inverse_loss,
    )

    # init pert model
    if args.perturbation_value_type == "zero":
        pert_val = torch.zeros(input_dim)
    elif args.perturbation_value_type == "one":
        ds = head_trainer.build_datasets(
            args.head_trainer.data_folder, args.head_trainer.train_set_ratio
        )[0]
        pert_val = torch.quantile(torch.stack([x[0] for x in ds]), q=0.8, dim=0)
    elif args.perturbation_value_type == "-one":
        ds = head_trainer.build_datasets(
            args.head_trainer.data_folder, args.head_trainer.train_set_ratio
        )[0]
        pert_val = torch.quantile(torch.stack([x[0] for x in ds]), q=0.2, dim=0)
    else:
        raise ValueError("Invalid perturbation value type")

    pert_model: PertModule = dispatch_pert_module(
        pert_type=args.pert.model_type,
        pert_num=args.pert.perturbation_num,
        pert_space=input_dim,
        pert_val=pert_val,
        pert_step=args.trainer.pert_num_steps,
        pert_config=args.pert,
    )

    # init trainer methods
    model = nn.Identity()

    # init surrogate and start perturbating
    trainer: SyntheticDataPerturbationTrainer
    match args.pert.model_type:
        case "subset_sampling":
            trainer = SyntheticDataPerturbationTrainer(
                pert_model,
                model,
                head_model,
                head_trainer.num_type_dict,
                args=args.trainer,
                has_wandb_writer=True,
            )
        case "sage":
            trainer = SyntheticDataSAGEPerturbationTrainer(
                pert_model,
                model,
                head_model,
                head_trainer.num_type_dict,
                args=args.trainer,
                has_wandb_writer=True,
            )
        case "fimap":
            trainer = SyntheticDataFIMAPPerturbationTrainer(
                pert_model,
                model,
                head_model,
                head_trainer.num_type_dict,
                args=args.trainer,
                has_wandb_writer=True,
            )
        case "lime":
            trainer = SyntheticDataLIMEPerturbationTrainer(
                pert_model,
                model,
                head_model,
                head_trainer.num_type_dict,
                args=args.trainer,
                has_wandb_writer=True,
            )
        case "cxplain":
            trainer = SyntheticDataCXPlainPerturbationTrainer(
                pert_model,
                model,
                head_model,
                head_trainer.num_type_dict,
                args=args.trainer,
                has_wandb_writer=True,
            )
        case "permutation":
            trainer = SyntheticDataPermutationPerturbationTrainer(
                pert_model,
                model,
                head_model,
                head_trainer.num_type_dict,
                args=args.trainer,
                has_wandb_writer=True,
            )
        case "saliency" | "approximation":
            trainer = SyntheticDataSaliencyPerturbationTrainer(
                pert_model,
                model,
                head_model,
                head_trainer.num_type_dict,
                args=args.trainer,
                has_wandb_writer=True,
            )
        case _:
            raise ValueError(f"Unknown perturbation model type: {args.pert.model_type}")

    trainer.train()


if __name__ == "__main__":
    main()
