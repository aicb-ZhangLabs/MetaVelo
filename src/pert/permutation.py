import torch

from typing import Callable, List, Dict, Literal
from dataclasses import dataclass
from .base import PertModule
from .nn.module import (
    BaseTriggerPerturbation,
    TriggerPerturbation,
    EmbeddingTriggerPerturbation,
)
from tqdm import tqdm


@dataclass
class PermutationPertModuleConfig:
    # trigger pert type
    trigger_pert_type: str = "trigger_perturbation"

    # strategy
    strategy: Literal["perturb", "mean", "ablation", "permutation"] = "perturb"

    permutation_iter_num: int = 16


class PermutationPertModule(PertModule):
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
        self.trigger_pert_type = kwargs["trigger_pert_type"]
        self.strategy = kwargs["strategy"]
        self.permutation_iter_num = kwargs["permutation_iter_num"]

        self.trigger_model = self._dispatch_trigger_perturbation(
            self.trigger_pert_type,
            pert_num,
            pert_space,
            pert_val,
            embedding_module=embedding_module,
            init_pert_ind=torch.LongTensor(init_pert_ind) if init_pert_ind else None,
            trigger_perturbation_kwargs=kwargs,
        )

        self.register_buffer("permutation_values", torch.zeros((pert_space,)))

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

    def _perturb_feature_update(
        self, x: torch.Tensor, y: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        forward_fn: Callable = kwargs["forward_fn"]
        pert_val = None
        match self.strategy:
            case "perturb":
                pert_val = self.pert_val
            case "mean":
                pert_val = torch.mean(x, dim=0)
            case "ablation":
                pert_val = torch.zeros_like(self.pert_val)
            case _:
                raise ValueError(f"Unknown strategy {self.strategy}")

        dummy_trigger_model = self._dispatch_trigger_perturbation(
            self.trigger_pert_type,
            1,
            self.pert_space,
            pert_val,
            embedding_module=self.embedding_module,
            init_pert_ind=torch.LongTensor([0]),
            trigger_perturbation_kwargs=kwargs,
        )
        dummy_trigger_model = dummy_trigger_model.to(x.device)

        permutation_values = torch.zeros_like(self.permutation_values)
        with torch.no_grad():
            for feat_ind in tqdm(
                range(self.pert_space), desc="Permutating", total=self.pert_space
            ):
                dummy_trigger_model.pert_ind.copy_(torch.LongTensor([feat_ind]))
                before_loss = torch.mean(forward_fn(x, y))
                after_loss = torch.mean(forward_fn(dummy_trigger_model(x), y))
                importances = after_loss - before_loss
                permutation_values[feat_ind] = importances.item()
        return permutation_values

    def _shuffle_feature_update(
        self, x: torch.Tensor, y: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        forward_fn: Callable = kwargs["forward_fn"]
        permutation_values = torch.zeros_like(self.permutation_values)
        with torch.no_grad():
            for feat_ind in tqdm(
                range(self.pert_space), desc="Permutating", total=self.pert_space
            ):
                importances = 0
                before_loss = torch.mean(forward_fn(x, y))
                for _ in range(self.permutation_iter_num):
                    x_perm = x.detach().clone()
                    x_perm[:, feat_ind] = x[torch.randperm(x_perm.size(0)), feat_ind]
                    after_loss = torch.mean(forward_fn(x_perm, y))
                    importances += after_loss

                permutation_values[feat_ind] = (
                    importances.item() / self.permutation_iter_num
                ) - before_loss.item()
        return permutation_values

    def get_importance_val(self, **kwargs) -> torch.Tensor:
        return self.permutation_values

    def update(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> None:
        """ref to scikit permutation

        :param x: permutation data
        :param y: permutation label, defaults to None
        """
        match self.strategy:
            case "perturb" | "mean" | "ablation":
                permutation_values = self.permutation_values = (
                    self._perturb_feature_update(x, y, **kwargs)
                )
            case "permutation":
                permutation_values = self.permutation_values = (
                    self._shuffle_feature_update(x, y, **kwargs)
                )
            case _:
                raise ValueError(f"Unknown strategy {self.strategy}")

        self.permutation_values.copy_(permutation_values)
        self.trigger_model.pert_ind.copy_(
            permutation_values.argsort(descending=True)[: self.pert_num]
        )
