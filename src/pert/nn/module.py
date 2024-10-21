import torch
import torch.nn as nn
import numpy as np

from typing import Dict
from abc import abstractmethod

from .op import (
    BatchTriggerPerturbationFunction,
    SequenceTriggerPerturbationFunction,
    SequenceEmbeddingTriggerPerturbationFunction,
    initial_random_ind,
    look_up_category_embedding,
)

EPSILON = np.finfo(np.float32).tiny


class BaseTriggerPerturbation(nn.Module):
    def __init__(
        self,
        pert_num: int,
        length: int,
        init_pert_ind: torch.LongTensor = None,
        init_pert_val: torch.Tensor = None,
    ):
        super().__init__()
        if pert_num > length:
            raise ValueError("Perturbation number is larger than the total length.")
        if length != init_pert_val.shape[-1]:
            raise ValueError("Perturbation value is incompactible the total length.")

        self.pert_num = pert_num
        self.length = length

        # if init_pert_vec is not given, we assume it would be loaded from a state dict
        self.register_buffer(
            "pert_ind",
            (
                init_pert_ind.detach().clone()
                if init_pert_ind is not None
                else torch.LongTensor(initial_random_ind(pert_num, length))
            ),
        )
        self.register_buffer(
            "pert_val",
            (
                init_pert_val.detach().clone().float()
                if init_pert_val is not None
                else torch.ones((self.length,))
            ),
        )

    @abstractmethod
    def forward(self, logits: torch.Tensor, tau: float = 1.0):
        raise NotImplementedError


class TriggerPerturbation(BaseTriggerPerturbation):
    def __init__(
        self,
        pert_num: int,
        length: int,
        init_pert_ind: torch.LongTensor = None,
        init_pert_val: torch.Tensor = None,
    ):
        super().__init__(pert_num, length, init_pert_ind, init_pert_val)

        self.weight = nn.Parameter(torch.zeros((pert_num, length)))

    def forward(self, x: torch.Tensor):
        pert_x = SequenceTriggerPerturbationFunction.apply(
            x, self.pert_ind, self.pert_val, self.weight
        )
        return pert_x

    @torch.inference_mode()
    def trial(self, x: torch.Tensor, pert_vecs: torch.Tensor):
        pert_x = BatchTriggerPerturbationFunction.apply(x, pert_vecs, self.pert_val)
        return pert_x


class EmbeddingTriggerPerturbation(TriggerPerturbation):
    def __init__(
        self,
        pert_num: int,
        length: int,
        embedding_module: nn.Module,
        init_pert_ind: torch.LongTensor = None,
        init_pert_val: torch.Tensor = None,
    ):
        super().__init__(pert_num, length, init_pert_ind, init_pert_val)
        self.register_buffer("embed_weight", embedding_module.weight.detach().clone())

    def forward(self, x: torch.Tensor):
        pert_x_embed = SequenceEmbeddingTriggerPerturbationFunction.apply(
            x, self.pert_ind, self.pert_val, self.embed_weight, self.weight
        )
        return pert_x_embed

    @torch.inference_mode()
    def trial(self, x: torch.Tensor, pert_vecs: torch.Tensor):
        pert_x = BatchTriggerPerturbationFunction.apply(x, pert_vecs, self.pert_val)

        pert_x_embed = look_up_category_embedding(pert_x, self.embed_weight)
        return pert_x_embed


class BaseSubsetSampling(nn.Module):
    def __init__(self, k, hard=False):
        super().__init__()
        self.k = k
        self.hard = hard

    @abstractmethod
    def forward(self, logits: torch.Tensor, tau: float = 1.0):
        raise NotImplementedError

    def straight_through(self, khot: torch.Tensor):
        """Generate straight through estimation of k-hot vector.

        :param khot: khot distribution [pert_space,]
        :return: hard khot vector [pert_space,]
        """
        # will do straight through estimation if training
        khot_hard = torch.zeros_like(khot)
        _, ind = torch.topk(khot, self.k, dim=-1)
        khot_hard = khot_hard.scatter_(-1, ind, 1)
        res = khot_hard - khot.detach() + khot
        return res

    def discrete_on_eval(self, khot: torch.Tensor):
        if self.training:
            return khot
        else:
            return self.straight_through(khot)


class SubsetSampling(BaseSubsetSampling):
    """This module is inspired from IJCAI 2019 paper: "Learning to Sample: An Active Learning Framework for Deep Object Detection".

    This module generate exactly k-hot vector.
    """

    def __init__(self, k, hard=False):
        super().__init__(k, hard)

    def forward(self, logits: torch.Tensor, tau: float = 1.0):
        log_scores = torch.log_softmax(logits, dim=-1)
        if self.training:
            gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.zeros_like(log_scores), torch.ones_like(log_scores)
            )
            g = gumbel_dist.sample()
        else:
            g = torch.zeros_like(log_scores)
            tau = 1.0

        # apply gumbel noise
        keys = log_scores + g

        # continuous top k
        khot = torch.zeros_like(keys)
        epsilon = keys.new_tensor([EPSILON])
        onehot_approx = torch.zeros_like(keys)
        for _ in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, epsilon)
            keys = keys + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(keys / tau, dim=-1)
            khot = khot + onehot_approx

        if self.hard:
            # will do straight through estimation if training
            pert_vec = self.straight_through(khot)
        else:
            pert_vec = self.discrete_on_eval(khot)

        return pert_vec, khot


class SubsetTopKSampling(BaseSubsetSampling):
    """This module is inspired from ICML 2018 paper: "Learning to Explain: An Information-Theoretic Perspective on Model Interpretation".

    https://github.com/Jianbo-Lab/L2X/blob/13fb3ad8f39e0b63180e933f9c60648dad6bc3db/imdb-word/explain.py#L188

    This module generate atmost k-hot vector.
    """

    def __init__(self, k, hard=False):
        super().__init__(k, hard)

    def forward(self, logits: torch.Tensor, tau: float = 1.0):
        log_scores = torch.log_softmax(logits, dim=-1)
        if self.training:
            gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.zeros_like(log_scores), torch.ones_like(log_scores)
            )
            g = gumbel_dist.sample((self.k,))
            keys = (log_scores.unsqueeze(0) + g) / tau  # [k, pert_space]
            khot = torch.max(torch.softmax(keys, dim=-1), dim=0)[0]
        else:
            khot = log_scores.exp()

        if self.hard:
            # will do straight through estimation if training
            pert_vec = self.straight_through(khot)
        else:
            pert_vec = self.discrete_on_eval(khot)
        return pert_vec, khot


class ReinMaxTopKSampling(BaseSubsetSampling):
    """This module is inspired from NeurIPS 2023 paper:

    https://github.com/microsoft/ReinMax/blob/main/reinmax/reinmax.py

    This module generate atmost k-hot vector.
    """

    def __init__(self, k, hard=True):
        if not hard:
            raise ValueError("ReinMaxTopKSampling only support hard sampling.")
        super().__init__(k, hard)

    def get_k_of_one_hots(
        self, sampled_one_hot_inds: torch.Tensor, categorial_num: int
    ) -> torch.Tensor:
        k_of_one_hot = sampled_one_hot_inds.new_zeros(
            (self.k, categorial_num), dtype=torch.float
        )
        row_inds = torch.arange(self.k, device=sampled_one_hot_inds.device)
        k_of_one_hot[row_inds, sampled_one_hot_inds] = 1.0
        return k_of_one_hot

    def forward(self, logits: torch.Tensor, tau: float = 1.0):
        scores = torch.softmax(logits, dim=-1)
        if self.training:
            sampled_one_hot_inds = torch.multinomial(scores, self.k, replacement=True)
            k_of_one_hot = self.get_k_of_one_hots(
                sampled_one_hot_inds, categorial_num=logits.shape[-1]
            )

            k_of_logits = logits.unsqueeze(0)
            pi1 = 0.5 * (k_of_one_hot + torch.softmax(k_of_logits / tau, dim=-1))
            pi1 = torch.softmax(
                (torch.log(pi1) - k_of_logits).detach() + k_of_logits, dim=-1
            )
            pi2 = 2 * pi1 - 0.5 * scores.unsqueeze(0)
            # khot = pi2.sum(0)
            khot = torch.max(pi2, dim=0)[0]
        else:
            sampled_one_hot_inds = torch.topk(scores, self.k, dim=-1)[1]
            k_of_one_hot = self.get_k_of_one_hots(
                sampled_one_hot_inds, categorial_num=logits.shape[-1]
            )
            khot = torch.zeros_like(scores)

        pert_vec = (
            k_of_one_hot.sum(0).greater_equal_(1.0).float() + khot - khot.detach()
        )
        return pert_vec, khot


class BaseSubsetSamplingPerturbation(nn.Module):
    sampler_dict: Dict[str, BaseSubsetSampling] = {
        "subset": SubsetSampling,
        "subset_topk": SubsetTopKSampling,
        "reinmax_topk": ReinMaxTopKSampling,
    }

    def __init__(
        self,
        pert_num: int,
        pert_space: int,
        tau=1.0,
        init_pert_val: torch.Tensor = None,
        hard=False,
        sampler_type: str = "subset",
    ):
        super().__init__()
        if pert_num > pert_space:
            raise ValueError("Perturbation number is larger than the total space.")

        self.pert_num = pert_num
        self.pert_space = pert_space
        self.subset_sampler = self._dispatch_sampler(sampler_type, hard)

        self.register_buffer("tau", torch.tensor(tau))
        self.register_buffer(
            "pert_val",
            (
                init_pert_val.detach().clone().float()
                if init_pert_val is not None
                else torch.ones((pert_space,))
            ),
        )

    def _dispatch_sampler(self, sampler_type: str, hard: bool) -> BaseSubsetSampling:
        if sampler_type in self.sampler_dict:
            return self.sampler_dict[sampler_type](self.pert_num, hard)
        else:
            raise ValueError("Unknown sampler type.")

    def _perturb(self, x: torch.Tensor, pert_vec: torch.Tensor) -> torch.Tensor:
        pert_vec = pert_vec.unsqueeze(0)
        pert_val = self.pert_val.unsqueeze(0)
        pert_x = (
            x + (pert_val - x) * pert_vec
        )  # x * (1 - pert_vec) + pert_vec * pert_val
        return pert_x

    @torch.no_grad()
    def update_tau(self, tau: float):
        self.tau.copy_(torch.tensor(tau))

    def get_weights(self):
        raise NotImplementedError

    def get_pert_vec(self):
        raise NotImplementedError


class SubsetSamplingPerturbation(BaseSubsetSamplingPerturbation):
    def __init__(
        self,
        pert_num: int,
        pert_space: int,
        tau=1.0,
        init_pert_val: torch.Tensor = None,
        hard=False,
        sampler_type: str = "subset",
        *,
        init_weights: torch.Tensor = None,
    ):
        super().__init__(
            pert_num=pert_num,
            pert_space=pert_space,
            tau=tau,
            init_pert_val=init_pert_val,
            hard=hard,
            sampler_type=sampler_type,
        )

        # sampling distribution weights
        self.weight = nn.Parameter(torch.zeros((pert_space,)))
        if init_weights is None:
            nn.init.uniform_(self.weight, -0.01, 0.01)
        else:
            with torch.no_grad():
                self.weight.copy_(init_weights)

    def forward(self, x):
        pert_vec, _ = self.subset_sampler(self.weight, tau=self.tau)

        pert_x = self._perturb(x, pert_vec)
        return pert_x


class ParameterizedSubsetSamplingPerturbation(BaseSubsetSamplingPerturbation):
    def __init__(
        self,
        pert_num: int,
        pert_space: int,
        tau=1.0,
        init_pert_val: torch.Tensor = None,
        hard=False,
        sampler_type: str = "subset",
        *,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__(
            pert_num,
            pert_space,
            tau,
            init_pert_val,
            hard=hard,
            sampler_type=sampler_type,
        )

        # prior logits
        # for more details: https://github.com/XiangLi1999/PrefixTuning/blob/6519d30e69b15a180f23e2cd41b766d3f62b8e82/gpt2/train_control.py#L155
        self.prior_embedding = nn.Embedding(pert_space, embedding_dim)
        self.prior_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_weights(self):
        prior_logits = self.prior_mlp(self.prior_embedding.weight).squeeze(-1)
        return prior_logits

    def get_pert_vec(self):
        prior_logits = self.get_weights()
        pert_vec, khot = self.subset_sampler(prior_logits, tau=self.tau)
        return pert_vec, khot

    def forward(self, x):
        pert_vec, _ = self.get_pert_vec()

        pert_x = self._perturb(x, pert_vec)
        return pert_x
