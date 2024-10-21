import torch

from typing import Callable, List, Dict
from dataclasses import dataclass
from torch.distributions import Normal
from .base import PertModule
from .nn.module import (
    BaseTriggerPerturbation,
    TriggerPerturbation,
    EmbeddingTriggerPerturbation,
)


@dataclass
class SaliencyPertModuleConfig:
    smooth_input: bool = False

    smooth_number: int = 32

    smooth_sigma: float = 1

    use_abs_grad: bool = False

    # trigger pert type
    trigger_pert_type: str = "trigger_perturbation"


class SaliencyPertModule(PertModule):
    def __init__(
        self,
        pert_num: int,
        pert_space: int,
        pert_val: torch.Tensor,
        embedding_module: torch.nn.Module = None,
        init_pert_ind: List[int] = [],
        **kwargs,
    ):
        super().__init__()
        self.pert_num = pert_num
        self.pert_space = pert_space
        self.pert_val = pert_val
        self.embedding_module = embedding_module
        self.smooth_input = kwargs["smooth_input"]
        self.smooth_number = kwargs["smooth_number"]
        self.smooth_sigma = kwargs["smooth_sigma"]
        self.use_abs_grad = kwargs["use_abs_grad"]
        self.trigger_pert_type = kwargs["trigger_pert_type"]

        self.trigger_model = self._dispatch_trigger_perturbation(
            self.trigger_pert_type,
            pert_num,
            pert_space,
            pert_val,
            embedding_module=embedding_module,
            init_pert_ind=torch.LongTensor(init_pert_ind) if init_pert_ind else None,
            trigger_perturbation_kwargs=kwargs,
        )

        self.register_buffer("saliency_values", torch.zeros((pert_space,)))
        self.register_buffer("saliency_number", torch.zeros((1,)))

    def _dispatch_trigger_perturbation(
        self,
        trigger_pert_type: str,
        pert_num: int,
        pert_space: int,
        pert_val: torch.Tensor,
        embedding_module: torch.nn.Module = None,
        init_pert_ind: torch.Tensor = None,
        trigger_perturbation_kwargs={},
    ) -> BaseTriggerPerturbation:
        if trigger_pert_type == "trigger_perturbation":
            return TriggerPerturbation(
                pert_num,
                pert_space,
                init_pert_val=pert_val,
                init_pert_ind=init_pert_ind,
            )
        elif trigger_pert_type == "embedding_trigger_perturbation":
            return EmbeddingTriggerPerturbation(
                pert_num,
                pert_space,
                embedding_module=embedding_module,
                init_pert_val=pert_val,
                init_pert_ind=init_pert_ind,
            )
        else:
            raise ValueError(f"Unknown trigger perturbation type {trigger_pert_type}")

    def to(self, device: torch.device):
        super().to(device)
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            "trigger_model": self.trigger_model.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.trigger_model.load_state_dict(state_dict["trigger_model"])

    def perturb(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.trigger_model(x)

    def get_pert_ind(self, **kwargs) -> torch.Tensor:
        return self.trigger_model.get_buffer("pert_ind")

    def get_importance_val(self, **kwargs) -> torch.Tensor:
        return self.saliency_values / self.saliency_number

    def update(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> None:
        """ref to scikit permutation

        :param x: permutation data
        :param y: permutation label, defaults to None
        """
        forward_fn: Callable = kwargs["forward_fn"]

        # create a new copy
        x = x.detach().clone()
        grad = None
        batch_sz = x.size(0)
        if self.smooth_input:
            gaussian_dist = Normal(0.0, self.smooth_sigma)
            noise_seq = gaussian_dist.sample(
                (batch_sz, self.smooth_number, x.size(1))
            ).to(x.device)
            x = x.unsqueeze(1) + noise_seq
            x.requires_grad_()
            loss = forward_fn(x.reshape((-1, x.size(2))))
            loss.backward()
            grad = (
                x.grad.reshape((batch_sz, self.smooth_number, -1))
                .mean(dim=1)
                .sum(dim=0)
            )
        else:
            x.requires_grad_()
            loss = forward_fn(x)
            loss.backward()
            grad = x.grad.sum(dim=0)

        # compute importance score
        importances = grad if not self.use_abs_grad else grad.abs()
        self.saliency_values.add_(importances)
        self.saliency_number.add_(torch.ones_like(self.saliency_number) * batch_sz)

        importances = self.saliency_values / self.saliency_number
        self.trigger_model.pert_ind.copy_(
            importances.argsort(descending=True)[: self.pert_num]
        )
