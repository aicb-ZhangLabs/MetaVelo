import torch

from torchdyn.core import NeuralODE
from dataclasses import asdict, dataclass, field
from torch.utils.data import DataLoader
from src.model.ode import RNAVeloNet, RNAVeloNetConfig
from src.model.classifier import FFNClassifier
from src.train.cell_type_cls import (
    CellTypeClassifierTrainerConfig,
)
from src.trainer.metavelo import PerturbationTrainer, PerturbationTrainerConfig
from src.wandb import WandbConfig, init_wandb
from src.slurm import SlurmConfig, slurm_launcher
from src.pert import PertModuleConfig, PertModule
from src.dispatch import dispatch_pert_module
from src.head.cls import NeuralClassifierHead
from src.utils import create_pert_val, seed_everything
from src.dataset import GeneTrajDataset


@dataclass
class ExperimentArgs:
    # trainer checkpoint path
    trainer_ckpt_path: str

    # eval data folder
    eval_data_folder: str

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


@slurm_launcher(ExperimentArgs)
def main(args: ExperimentArgs):
    seed_everything(args.trainer.seed)
    init_wandb(args.wandb, asdict(args))

    # init head model
    head_layer = FFNClassifier(
        args.model.input_dim,
        len(args.head_trainer.cell_type_list),
        hidden_dim=args.head_trainer.hidden_dim,
    )

    head_model = NeuralClassifierHead(
        head_layer,
        tgt_ind=args.head_trainer.cell_type_list.index(args.head_trainer.tgt_cell_type),
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
    trainer: PerturbationTrainer
    trainer = PerturbationTrainer(
        pert_model, model, head_model, args=args.trainer, has_wandb_writer=True
    )

    # load perturbation trainer state
    state_pt = torch.load(args.trainer_ckpt_path)
    trainer.load_state(state_pt)

    # run evaluation
    eval_ds = GeneTrajDataset(
        args.eval_data_folder, starting_cell_type=args.trainer.starting_cell_type
    )
    eval_dl = DataLoader(
        eval_ds, batch_size=args.trainer.train_batch_size, shuffle=False
    )
    trainer.eval(eval_dl)


if __name__ == "__main__":
    main()
