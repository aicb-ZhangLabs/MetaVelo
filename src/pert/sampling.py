import torch
import torch.optim as optim

from typing import Any, Dict
from dataclasses import dataclass, field
from .base import PertModule
from .nn.module import (
    BaseSubsetSamplingPerturbation,
    SubsetSamplingPerturbation,
    ParameterizedSubsetSamplingPerturbation,
)


@dataclass
class SubsetSamplingPertModuleConfig:
    # sampling optimizer
    optimizer_name: str = "Adam"

    # sampling optimizer lr
    lr: float = 1e-4

    # use lr scheduler
    use_scheduler: bool = True

    # sampling optimizer weight decay
    optimizer_kwargs: Dict = field(
        default_factory={
            "Adam": {"betas": (0.9, 0.99)},
            "SGD": {"momentum": 0.0},
        }.copy
    )

    # sampling gumble final temperature
    tau: float = 0.05

    # sampling gumble starting temperature
    tau_start: float = 5.0

    # sampling gumble anneal steps
    tau_anneal_steps: int = 1000

    # sampling gumble anneal scheduler
    tau_scheduler_name: str = "linear"

    # hard sampling
    hard: bool = True

    # subset perturbation module type
    sampler_pert_type: str = "subset_sampling"

    # subsuet sampler type
    sampler_type: str = "subset"

    # parameterized subset perturbation module embedding dim
    parameterized_embedding_dim: int = 32

    # parameterized subset perturbation module hidden dim
    parameterized_hidden_dim: int = 64


class SubsetSamplingPertModule(PertModule):
    sampler_pert_dict: Dict[str, BaseSubsetSamplingPerturbation] = {
        "subset_sampling": SubsetSamplingPerturbation,
        "parameterized_subset_sampling": ParameterizedSubsetSamplingPerturbation,
    }

    def __init__(
        self,
        pert_num: int,
        pert_space: int,
        pert_val: torch.Tensor,
        pert_step: int,
        embedding_module: torch.nn.Module = None,
        tau: float = 0.05,
        tau_start: float = 5.0,
        tau_anneal_steps: int = 1000,
        tau_scheduler_name: str = "linear",
        hard: bool = False,
        optimizer_name: str = "Adam",
        lr: float = 1e-3,
        use_scheduler: bool = True,
        optimizer_kwargs: Dict[str, Dict[str, Any]] = {},
        **kwargs,
    ):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.optimizer_kwargs = optimizer_kwargs
        self.embedding_module = embedding_module

        self.pert_step = pert_step
        self.tau = tau
        self.tau_start = tau_start
        self.tau_anneal_steps = tau_anneal_steps
        self.tau_scheduler_name = tau_scheduler_name

        self.sampler_model = self._dispatch_sampling_perturbation(
            kwargs["sampler_pert_type"],
            pert_num,
            pert_space,
            tau=tau,
            init_pert_val=pert_val,
            hard=hard,
            sampler_type=kwargs["sampler_type"],
            sampler_perturbation_kwargs=kwargs,
        )
        self.sampler_opt, self.sampler_opt_scheduler = self._create_optim(
            self.sampler_model,
            optimizer_name,
            use_scheduler,
            lr,
            optimizer_kwargs,
            total_step=self.pert_step,
        )

    def _dispatch_sampling_perturbation(
        self,
        sampler_pert_type: str,
        pert_num: int,
        pert_space: int,
        tau=1.0,
        init_pert_val: torch.Tensor = None,
        hard=False,
        sampler_type: str = "subset",
        sampler_perturbation_kwargs={},
    ) -> BaseSubsetSamplingPerturbation:
        if sampler_pert_type == "subset_sampling":
            return SubsetSamplingPerturbation(
                pert_num=pert_num,
                pert_space=pert_space,
                tau=tau,
                init_pert_val=init_pert_val,
                hard=hard,
                sampler_type=sampler_type,
            )
        elif sampler_pert_type == "parameterized_subset_sampling":
            return ParameterizedSubsetSamplingPerturbation(
                pert_num=pert_num,
                pert_space=pert_space,
                tau=tau,
                init_pert_val=init_pert_val,
                hard=hard,
                sampler_type=sampler_type,
                embedding_dim=sampler_perturbation_kwargs[
                    "parameterized_embedding_dim"
                ],
                hidden_dim=sampler_perturbation_kwargs["parameterized_hidden_dim"],
            )
        else:
            raise ValueError(f"Unknown sampler perturbation type: {sampler_pert_type}")

    @staticmethod
    def _create_optim(
        sampler_model: BaseSubsetSamplingPerturbation,
        optimizer_name: str,
        use_scheduler: bool,
        lr: float,
        optimizer_kwargs: Dict[str, Dict[str, Any]],
        total_step: int = 100,
    ) -> optim.Optimizer:
        opt_cls = getattr(optim, optimizer_name)
        opt = opt_cls(
            [
                {
                    "params": sampler_model.parameters(),
                    "lr": lr,
                },
            ],
            **optimizer_kwargs.get(optimizer_name, {}),
        )

        if use_scheduler:
            opt_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_step)
        else:
            # here we use constant lr scheduler to let the lr be constant
            opt_scheduler = optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        return opt, opt_scheduler

    def to(self, device: torch.device):
        super().to(device)

        # correctly create optimizer after moving the module to device
        self.sampler_opt, self.sampler_opt_scheduler = self._create_optim(
            self.sampler_model,
            self.optimizer_name,
            self.use_scheduler,
            self.lr,
            self.optimizer_kwargs,
            total_step=self.pert_step,
        )
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            "sampler_model": self.sampler_model.state_dict(),
            "sampler_opt": self.sampler_opt.state_dict(),
            "sampler_opt_scheduler": self.sampler_opt_scheduler.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.sampler_model.load_state_dict(state_dict["sampler_model"])
        self.sampler_opt.load_state_dict(state_dict["sampler_opt"])
        self.sampler_opt_scheduler.load_state_dict(state_dict["sampler_opt_scheduler"])

    def get_tau_scheduler(self, scheduler_name: str, **scheduler_kwargs):
        if scheduler_name == "constant":
            return lambda step: self.tau
        elif scheduler_name == "linear":
            return lambda step: self.tau_start - (self.tau_start - self.tau) * min(
                1.0, step / self.tau_anneal_steps
            )
        else:
            raise ValueError(f"Unknown scheduler name: {scheduler_name}")

    def perturb(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # only update tau during training stage
        update_tau = kwargs.get("update_tau", False)
        if update_tau:
            tau = self.get_tau_scheduler(self.tau_scheduler_name)(kwargs["step"])
            self.sampler_model.update_tau(tau)
        return self.sampler_model(x)

    def get_pert_ind(self, **kwargs) -> torch.Tensor:
        pert_vec, _ = self.sampler_model.get_pert_vec()
        return torch.where(pert_vec)[0]

    def get_importance_val(self, **kwargs) -> torch.Tensor:
        return self.sampler_model.get_weights()

    def update(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> None:
        self.sampler_opt.step()
        self.sampler_opt.zero_grad()
        self.sampler_opt_scheduler.step()
