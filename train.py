import torch

from torchdyn.core import NeuralODE
from src.model.ode import RNAVeloNet, RNAVeloNetConfig
from src.trainer.metavelo import Trainer, TrainerConfig
from src.wandb import WandbConfig, init_wandb
from src.slurm import SlurmConfig, slurm_launcher
from dataclasses import asdict, dataclass, field
from src.utils import seed_everything


@dataclass
class ExperimentConfig:
    # model configurations
    model: RNAVeloNetConfig = field(default_factory=RNAVeloNetConfig)

    # trainer arguments
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # wandb
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # slurm
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # trainer should be resumed
    resume_trainer: bool = False

    # trainer checkpoint path
    trainer_ckpt_path: str = ""


@slurm_launcher(ExperimentConfig)
def main(args: ExperimentConfig):
    seed_everything(args.trainer.seed)
    init_wandb(args.wandb, asdict(args))

    field = RNAVeloNet(args.model)
    model = NeuralODE(
        field,
        sensitivity="adjoint",
        solver="rk4",
        solver_adjoint="dopri5",
        atol_adjoint=1e-4,
        rtol_adjoint=1e-4,
    )

    trainer = Trainer(model, args=args.trainer, has_wandb_writer=True)
    if args.resume_trainer:
        state_pt = torch.load(args.trainer_ckpt_path, map_location=trainer.device)
        trainer.load_state(state_pt)

    trainer.train()


if __name__ == "__main__":
    main()
