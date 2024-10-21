import torch
import torch.nn as nn

from typing import Dict
from abc import abstractmethod


class HeadModule(nn.Module):
    @abstractmethod
    def get_loss(
        self, x: torch.Tensor, target: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """compute loss function for the head module

        this function should be broadcastable for the input tensor x
        x, target


        :param x: input tensor
        :raises NotImplementedError: all head module should implement this method
        :return: a loss scaler value or a tensor of loss values for each input
        """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        raise NotImplementedError
