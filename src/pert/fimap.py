import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from typing import Any, Dict, Literal
from dataclasses import dataclass, field
from .base import PertModule
from .nn.op import initial_random_ind, create_pert_vec
from ..model.ae import ATACSeqAEModel


class FIMAPPerturbation(nn.Module):
    def __init__(
        self,
        pert_num: int,
        pert_space: int,
        init_pert_val: torch.Tensor = None,
        tau: float = 0.05,
        weight_model: str = "mlp",
        sampling_method: str = "smooth",
    ) -> None:
        super().__init__()
        if pert_num > pert_space:
            raise ValueError("Perturbation number is larger than the total space.")

        self.pert_num = pert_num
        self.pert_space = pert_space
        self.tau = tau
        self.sampling_method = sampling_method

        self.register_buffer(
            "pert_val",
            (
                init_pert_val.detach().clone().float()
                if init_pert_val is not None
                else torch.ones((pert_space,))
            ),
        )

        # perturbation weights
        if weight_model == "mlp":
            self.weight_net = nn.Sequential(
                nn.Linear(pert_space, 128),
                nn.BatchNorm1d(128),
                nn.PReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.PReLU(),
                nn.Linear(64, 64),
                nn.PReLU(),
                nn.Linear(64, pert_space),
            )
        elif weight_model == "split_ae":
            self.weight_net = ATACSeqAEModel(
                self.pert_space,
                chromosome_dims=[
                    11348,
                    6169,
                    6997,
                    6364,
                    3554,
                    4233,
                    4076,
                    4178,
                    4966,
                    3189,
                    3807,
                    10272,
                    3469,
                    1478,
                    2488,
                    8367,
                    6290,
                    7655,
                    7203,
                    6950,
                    6250,
                    5547,
                    2245,
                    124,
                ],
                hidden_dims=[16, 32],
                latent_dim=20,
            )
        else:
            raise ValueError(f"Invalid weight model: {weight_model}")

    @torch.inference_mode()
    def get_pert_ind(self, x) -> torch.Tensor:
        weight = self.weight_net(x)
        weight = torch.sigmoid(weight)
        return weight

    def _get_pert_vec(self, weight):
        if self.sampling_method == "smooth":
            pert_vec = torch.sigmoid(weight)
        elif self.sampling_method == "gumbel":
            # dist = RelaxedBernoulli(self.tau, logits=weight)
            # pert_prob = dist.rsample()
            # pert_vec = pert_prob.round() + (pert_prob - pert_prob.detach())

            pert_prob = torch.sigmoid(weight)
            pert_vec = pert_prob.round() + (pert_prob - pert_prob.detach())
        else:
            raise ValueError(f"Invalid sampling method: {self.sampling_method}")
        return pert_vec

    def forward(self, x):
        weight = self.weight_net(x)
        pert_vec = self._get_pert_vec(weight)
        pert_x = self._perturb(x, pert_vec)
        return pert_x

    def _perturb(self, x: torch.Tensor, pert_vec: torch.Tensor) -> torch.Tensor:
        pert_val = self.pert_val.unsqueeze(0)
        pert_x = (
            x + (pert_val - x) * pert_vec
        )  # x * (1 - pert_vec) + pert_vec * pert_val
        return pert_x


# class FIMAPEmbeddingPerturbation(FIMAPPerturbation):
#     def __init__(
#         self,
#         pert_num: int,
#         pert_space: int,
#         embedding_module: torch.nn.Module,
#         init_pert_val: torch.Tensor = None,
#         init_weights: torch.Tensor = None,
#         tau: float = 0.05,
#     ) -> None:
#         super().__init__(pert_num, pert_space, init_pert_val, init_weights, tau)
#         self.register_buffer("embed_weight", embedding_module.weight.detach().clone())

#     def _perturb(self, x: torch.Tensor, pert_vec: torch.Tensor) -> torch.Tensor:
#         pert_x = super()._perturb(x, pert_vec)
#         return look_up_category_embedding(pert_x, self.embed_weight)


@dataclass
class FIMAPPertModuleConfig:
    # sampling gumble final temperature
    tau: float = 0.05

    # sampling gumble starting temperature
    tau_start: float = 5.0

    # sampling gumble anneal steps
    tau_anneal_steps: int = 1000

    # sampling gumble anneal scheduler
    tau_scheduler_name: str = "linear"

    # weight model
    weight_model: Literal["mlp", "split_ae"] = "mlp"

    sampling_method: Literal["smooth", "gumbel"] = "smooth"

    regularization_weight: float = 0.0

    # sampling optimizer
    optimizer_name: str = "Adam"

    # sampling optimizer lr
    lr: float = 1e-4

    # sampling optimizer weight decay
    optimizer_kwargs: Dict = field(
        default_factory={
            "Adam": {"betas": (0.9, 0.99)},
            "SGD": {"momentum": 0.0},
        }.copy
    )


class FIMAPPertModule(PertModule):
    def __init__(
        self,
        pert_num: int,
        pert_space: int,
        pert_val: torch.Tensor,
        embedding_module: torch.nn.Module = None,
        weight_model: str = "mlp",
        tau: float = 0.05,
        optimizer_name: str = "Adam",
        lr: float = 1e-3,
        optimizer_kwargs: Dict[str, Dict[str, Any]] = {},
        regularization_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.embedding_module = embedding_module
        self.tau = tau
        self.tau_start = kwargs["tau_start"]
        self.tau_anneal_steps = kwargs["tau_anneal_steps"]
        self.tau_scheduler_name = kwargs["tau_scheduler_name"]
        self.weight_model = weight_model
        self.sampling_method = kwargs["sampling_method"]

        self.sampler_model = FIMAPPerturbation(
            pert_num,
            pert_space,
            init_pert_val=pert_val,
            weight_model=weight_model,
            sampling_method=self.sampling_method,
        )
        self.sampler_opt = self._create_optim(
            self.sampler_model,
            optimizer_name,
            lr,
            optimizer_kwargs,
        )
        self.regularization_weight = regularization_weight

        self.register_buffer(
            "pert_ind",
            torch.LongTensor(initial_random_ind(pert_num, pert_space)),
        )

    @staticmethod
    def _create_optim(
        sampler_model: FIMAPPerturbation,
        optimizer_name: str,
        lr: float,
        optimizer_kwargs: Dict[str, Dict[str, Any]],
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
        return opt

    def to(self, device: torch.device):
        super().to(device)

        # correctly create optimizer after moving the module to device
        self.sampler_opt = self._create_optim(
            self.sampler_model,
            self.optimizer_name,
            self.lr,
            self.optimizer_kwargs,
        )
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            "sampler_model": self.sampler_model.state_dict(),
            "sampler_opt": self.sampler_opt.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.sampler_model.load_state_dict(state_dict["sampler_model"])
        self.sampler_opt.load_state_dict(state_dict["sampler_opt"])

    def perturb(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pert_ind = kwargs.get("pert_ind", None)
        if pert_ind is None:
            if self.sampling_method == "gumbel":
                # only update tau during training stage
                update_tau = kwargs.get("update_tau", False)
                if update_tau:
                    tau = self.get_tau_scheduler(self.tau_scheduler_name)(
                        kwargs["step"]
                    )
                    self.sampler_model.tau = tau

            return self.sampler_model(x)
        else:
            pert_vec = create_pert_vec(pert_ind, self.sampler_model.pert_space).to(
                x.device
            )
            return self.sampler_model._perturb(x, pert_vec)

    def get_tau_scheduler(self, scheduler_name: str, **scheduler_kwargs):
        if scheduler_name == "constant":
            return lambda step: self.tau
        elif scheduler_name == "linear":
            return lambda step: self.tau_start - (self.tau_start - self.tau) * min(
                1.0, step / self.tau_anneal_steps
            )
        else:
            raise ValueError(f"Unknown scheduler name: {scheduler_name}")

    @torch.inference_mode()
    def get_pert_ind(self, **kwargs) -> torch.Tensor:
        X = kwargs["x"]

        if self.sampling_method == "smooth":
            probs = self.sampler_model.get_pert_ind(X)
            indices = torch.mean(probs, dim=0).topk(self.sampler_model.pert_num).indices
        else:
            # per_indices = torch.zeros_like(X)
            # pert_inds = (
            #     self.sampler_model.get_pert_ind(X)
            #     .topk(self.sampler_model.pert_num)
            #     .indices
            # )
            # per_indices.scatter_(1, pert_inds, 1)
            # indices = (
            #     torch.mean(per_indices, dim=0).topk(self.sampler_model.pert_num).indices
            # )

            per_indices = self.sampler_model.get_pert_ind(X).round()
            indices = (
                torch.mean(per_indices, dim=0).topk(self.sampler_model.pert_num).indices
            )

        return indices

    def get_importance_val(self, **kwargs) -> torch.Tensor:
        X = kwargs["x"]
        if self.sampling_method == "smooth":
            probs = self.sampler_model.get_pert_ind(X)
            return torch.mean(probs, dim=0)
        else:
            # per_indices = torch.zeros_like(X)
            # pert_inds = (
            #     self.sampler_model.get_pert_ind(X)
            #     .topk(self.sampler_model.pert_num)
            #     .indices
            # )
            # per_indices.scatter_(1, pert_inds, 1)
            # indices = (
            #     torch.mean(per_indices, dim=0).topk(self.sampler_model.pert_num).indices
            # )

            per_indices = self.sampler_model.get_pert_ind(X).round()
            return torch.mean(per_indices, dim=0)

    def update(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> None:
        if kwargs.get("add_regularization", True):
            regu_loss = (
                F.l1_loss(
                    self.sampler_model(x),
                    x,
                    reduction="mean",
                )
                * self.regularization_weight
            )
            regu_loss.backward()

        self.sampler_opt.step()
        self.sampler_opt.zero_grad()
