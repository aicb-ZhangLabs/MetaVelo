import torch
import torch.nn as nn

from skorch import NeuralNet
from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from dataclasses import dataclass
from torch.utils.data import TensorDataset
from typing import Callable, Literal, Union
from tqdm import tqdm
from ..utils.mask import MaskInput
from .module import MLPModelBuilder, UNetModelBuilder, BalanceKLDivLoss


@torch.inference_mode()
def create_dataset(
    samples: torch.Tensor,
    targets: Union[torch.Tensor, None],
    forward_fn: Callable,
    head_loss_fn: Callable,
    mask_fn: MaskInput,
    mask_value: torch.Tensor,
) -> TensorDataset:
    list_of_predictions = []
    list_of_samples = torch.split(samples, 1, dim=0)
    list_of_targets = (
        torch.split(targets, 1, dim=0)
        if targets is not None
        else [None] * len(list_of_samples)
    )
    for sample, target in tqdm(
        zip(list_of_samples, list_of_targets),
        desc="creating cxplain dataset",
        total=len(list_of_samples),
    ):
        list_of_importances = []
        for masked_samples in mask_fn.mask_sample(sample, mask_value):
            # expand target to match the size of masked_samples
            expanded_target = None
            if target is not None:
                expanded_size = [masked_samples.size(0)] + [-1] * (target.dim() - 1)
                expanded_target = target.expand(*expanded_size)

            # compute the loss
            origin_loss = head_loss_fn(forward_fn(sample), target)
            masked_loss = head_loss_fn(forward_fn(masked_samples), expanded_target)
            importances = masked_loss - origin_loss
            list_of_importances.append(importances)

        importances = torch.cat(list_of_importances, dim=0)

        # details: /home/junhal11/miniconda3/envs/torchvelo/lib/python3.10/site-packages/cxplain/backend/causal_loss.py:get_delta_errors_fixed_size
        importances = torch.maximum(
            importances, importances.new_tensor(torch.finfo(importances.dtype).tiny)
        )

        normalized_importances = importances / torch.sum(
            importances, dim=0, keepdim=True
        )  # [N, 1]
        list_of_predictions.append(normalized_importances.flatten())

    predictions = torch.stack(list_of_predictions, dim=0)
    dataset = TensorDataset(samples, predictions)
    return dataset


@dataclass
class CXPlainTrainerConfig:
    model_type: Literal["mlp", "unet"] = "mlp"
    unet_conv_type: Literal["1d", "2d"] = "2d"
    batch_size: int = 128
    epoch: int = 500
    early_stopping_patience: int = 10
    learning_rate: float = 1e-3
    downsample_factors: int = 2
    num_layers: int = 2
    num_units: int = 64
    seed: int = 2024
    use_balance_kl_loss: bool = False


def train_cxplain(
    train_ds: TensorDataset,
    valid_ds: TensorDataset,
    device: torch.DeviceObjType,
    config: CXPlainTrainerConfig,
) -> torch.nn.Module:
    model_types = {
        "mlp": MLPModelBuilder,
        "unet": UNetModelBuilder,
    }

    early_stop_callback = EarlyStopping(
        patience=config.early_stopping_patience, threshold_mode="abs"
    )
    model = NeuralNet(
        module=model_types[config.model_type],
        module__downsample_factors=config.downsample_factors,
        module__num_layers=config.num_layers,
        module__num_units=config.num_units,
        module__conv_type=config.unet_conv_type,
        criterion=nn.KLDivLoss if not config.use_balance_kl_loss else BalanceKLDivLoss,
        criterion__reduction="batchmean",
        optimizer=torch.optim.Adam,
        optimizer__lr=config.learning_rate,
        train_split=predefined_split(valid_ds),
        batch_size=config.batch_size,
        max_epochs=config.epoch,
        device=device,
        callbacks=[early_stop_callback],
        iterator_train__shuffle=True,
    )

    model.fit(train_ds)
    torch_module = model.module_
    return torch_module
