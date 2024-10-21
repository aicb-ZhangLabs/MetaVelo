import torch

from typing import Literal
from dataclasses import dataclass, field, asdict
from .base import PertModule
from .trigger import TriggerPertModule, TriggerPertModuleConfig
from .sampling import SubsetSamplingPertModule, SubsetSamplingPertModuleConfig
from .sage import SAGEPertModule, SAGEPertModuleConfig
from .cxplain import CXPlainPertModule, CXPlainPertModuleConfig
from .lime import LIMEPertModule, LIMEPertModuleConfig
from .fimap import FIMAPPertModule, FIMAPPertModuleConfig
from .permutation import PermutationPertModule, PermutationPertModuleConfig
from .saliency import SaliencyPertModule, SaliencyPertModuleConfig
from .approximation import ApproximationPertModule, ApproximationPertModuleConfig


pert_dict = {
    "trigger": TriggerPertModule,
    "subset_sampling": SubsetSamplingPertModule,
    "sage": SAGEPertModule,
    "cxplain": CXPlainPertModule,
    "lime": LIMEPertModule,
    "fimap": FIMAPPertModule,
    "permutation": PermutationPertModule,
    "saliency": SaliencyPertModule,
    "approximation": ApproximationPertModule,
}


@dataclass
class PertModuleConfig:
    # pert model name
    model_type: Literal[
        "trigger", "subset_sampling", "sage", "cxplain", "lime", "fimap", "permutation"
    ] = "trigger"

    # perturbation num
    perturbation_num: int = 8

    # trigger perturbation configurations
    trigger: TriggerPertModuleConfig = field(default_factory=TriggerPertModuleConfig)

    # subset sampling perturbation configurations
    subset_sampling: SubsetSamplingPertModuleConfig = field(
        default_factory=SubsetSamplingPertModuleConfig
    )

    # trigger with sage perturbation configurations
    sage: SAGEPertModuleConfig = field(default_factory=SAGEPertModuleConfig)

    # trigger with cxplain perturbation configurations
    cxplain: CXPlainPertModuleConfig = field(default_factory=CXPlainPertModuleConfig)

    lime: LIMEPertModuleConfig = field(default_factory=LIMEPertModuleConfig)

    fimap: FIMAPPertModuleConfig = field(default_factory=FIMAPPertModuleConfig)

    permutation: PermutationPertModuleConfig = field(
        default_factory=PermutationPertModuleConfig
    )

    saliency: SaliencyPertModuleConfig = field(default_factory=SaliencyPertModuleConfig)

    approximation: ApproximationPertModuleConfig = field(
        default_factory=ApproximationPertModuleConfig
    )


def build_pert_module(
    pert_type: str,
    pert_num: int,
    pert_space: int,
    pert_val: torch.Tensor,
    pert_step: int,
    pert_config: PertModuleConfig,
    embedding_module: torch.nn.Module = None,
) -> PertModule:
    return pert_dict[pert_type](
        pert_num=pert_num,
        pert_space=pert_space,
        pert_val=pert_val,
        pert_step=pert_step,
        embedding_module=embedding_module,
        **asdict(getattr(pert_config, pert_type)),
    )
