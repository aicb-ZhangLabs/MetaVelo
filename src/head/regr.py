from typing import Dict
import torch
import torch.nn as nn

from .base import HeadModule
from typing import Union


class ValueRegressionHead(HeadModule):
    def __init__(
        self,
        ind: int,
        increase_value: bool = True,
        penalty_weight: Union[float, None] = None,
    ):
        super().__init__()
        self.ind = ind
        self.increase_value = increase_value
        self.penalty_weight = penalty_weight

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {}
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        pass

    def remove_index(self, tensor: torch.Tensor, dim: int, index: int):
        slices = []
        if index > 0:
            slices.append(tensor.narrow(dim, 0, index))
        if index < tensor.size(dim) - 1:
            slices.append(tensor.narrow(dim, index + 1, tensor.size(dim) - index - 1))

        return torch.cat(slices, dim)

    def get_loss(
        self, x: torch.Tensor, target: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """compute target loss

        :param x: input tensor
        :raises ValueError: this function requires target value
        :return: loss value
        """
        if target is None:
            raise ValueError("target is not provided")

        if x.size(0) != target.size(0):
            if x.size(0) % target.size(0) != 0:
                raise ValueError(
                    "the batch size of input should be divisible by target batch size"
                )
            else:
                # for batch trial implementation, the dimension 0 is always the batch size
                target = torch.tile(
                    target, (x.size(0) // target.size(0), *([1] * (target.dim() - 1)))
                )

        index = x.new_ones(1, dtype=torch.long) * self.ind
        ind_x = torch.index_select(x, dim=1, index=index).squeeze(dim=-1)
        ind_target = torch.index_select(target, dim=1, index=index).squeeze(dim=-1) + 1

        criterion = nn.MSELoss(**kwargs)
        if not self.increase_value:
            loss = criterion(ind_x / ind_target, torch.zeros_like(ind_target))
        else:
            loss = criterion(ind_x / ind_target, torch.ones_like(ind_target))

        # add penalty if necessary
        if self.penalty_weight is not None:
            unind_x = self.remove_index(x, 1, self.ind)
            unind_target = self.remove_index(target, 1, self.ind) + 1

            if criterion.reduction in {"mean", "sum"}:
                penalty = criterion(
                    unind_x / unind_target, torch.ones_like(unind_target)
                )
            else:
                # no reduction
                penalty = torch.mean(
                    criterion(unind_x / unind_target, torch.ones_like(unind_target)),
                    dim=-1,
                )
            loss += self.penalty_weight * penalty
        return loss
