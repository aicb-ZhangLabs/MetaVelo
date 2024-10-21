import torch

from torchdyn.core import NeuralODE
from dataclasses import asdict
from src.model.ode import RNAVeloNet
from src.model.classifier import FFNClassifier
from src.train.cell_type_cls import (
    CellTypeClassifierTrainer,
)
from src.trainer.metavelo import (
    PerturbationTrainer,
    SAGEPerturbationTrainer,
    FIMAPPerturbationTrainer,
    LIMEPerturbationTrainer,
    CXPlainPerturbationTrainer,
    PermutationPerturbationTrainer,
    SaliencyPerturbationTrainer,
)
from src.wandb import init_wandb
from src.slurm import slurm_launcher
from src.pert import PertModule
from src.dispatch import dispatch_pert_module
from src.head.cls import NeuralClassifierHead
from src.utils import create_pert_val, seed_everything
from configs.perturb.config import ConfiguredExperimentArgs, ExperimentArgs


@slurm_launcher(ConfiguredExperimentArgs)
def main(args: ExperimentArgs):
    seed_everything(args.trainer.seed)
    init_wandb(args.wandb, asdict(args))

    # train head model
    head_layer = FFNClassifier(
        args.model.input_dim,
        len(args.head_trainer.cell_type_list),
        hidden_dim=args.head_trainer.hidden_dim,
    )
    head_trainer = CellTypeClassifierTrainer(
        head_layer, args=args.head_trainer, has_wandb_writer=True
    )
    head_trainer.train()

    head_model = NeuralClassifierHead(
        head_layer,
        tgt_ind=args.head_trainer.cell_type_list.index(args.head_trainer.tgt_cell_type),
        inverse_loss=args.inverse_loss,
    )

    # init pert model
    pert_val = torch.from_numpy(create_pert_val(args.trainer.ann_prc_data)).to(
        torch.float32
    )
    pert_model: PertModule = dispatch_pert_module(
        pert_type=args.pert.model_type,
        pert_num=args.pert.perturbation_num,
        pert_space=args.model.input_dim,
        pert_val=pert_val,
        pert_step=args.trainer.pert_num_steps,
        pert_config=args.pert,
    )

    # init ode and start perturbating
    field = RNAVeloNet(args.model)
    model = NeuralODE(
        field,
        sensitivity="adjoint",
        solver="rk4",
        solver_adjoint="dopri5",
        atol_adjoint=1e-4,
        rtol_adjoint=1e-4,
    )

    # match trainer
    trainer: PerturbationTrainer
    match args.pert.model_type:
        case "subset_sampling":
            trainer = PerturbationTrainer(
                pert_model, model, head_model, args=args.trainer, has_wandb_writer=True
            )
        case "sage":
            trainer = SAGEPerturbationTrainer(
                pert_model, model, head_model, args=args.trainer, has_wandb_writer=True
            )
        case "fimap":
            trainer = FIMAPPerturbationTrainer(
                pert_model, model, head_model, args=args.trainer, has_wandb_writer=True
            )
        case "lime":
            trainer = LIMEPerturbationTrainer(
                pert_model, model, head_model, args=args.trainer, has_wandb_writer=True
            )
        case "cxplain":
            trainer = CXPlainPerturbationTrainer(
                pert_model, model, head_model, args=args.trainer, has_wandb_writer=True
            )
        case "permutation":
            trainer = PermutationPerturbationTrainer(
                pert_model, model, head_model, args=args.trainer, has_wandb_writer=True
            )
        case "saliency" | "approximation":
            trainer = SaliencyPerturbationTrainer(
                pert_model, model, head_model, args=args.trainer, has_wandb_writer=True
            )
        case _:
            raise ValueError(f"Unknown perturbation model type: {args.pert.model_type}")

    # trainer = PerturbationTrainer(
    #     pert_model, model, head_model, args=args.trainer, has_wandb_writer=True
    # )

    state_pt = torch.load(args.trainer_ckpt_path)
    if not args.resume_trainer:
        super(PerturbationTrainer, trainer).load_state(state_pt)
    else:
        trainer.load_state(state_pt)

    trainer.train()


if __name__ == "__main__":
    main()
