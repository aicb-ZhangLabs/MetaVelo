import sklearn
import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass, asdict
from skorch import NeuralNet
from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from src.dataset import load_deepvelo_dataset, load_raw_dataset
from src.model.ae import AEModelConfig, ShallowAEModel


@dataclass
class DeepveloConfig:
    # raw dataset name
    raw_dataset_name: str = None

    # raw dataset save path
    raw_dataset_path: str = None

    # read raw dataset from cache
    raw_dataset_use_cache: bool = False

    # deepvelo dataset path
    deepvelo_dataset_path: str = None

    # read deepvelo dataset from cache
    deepvelo_dataset_use_cache: bool = False

    # lr
    learning_rate: float = 5e-5

    # batch size
    batch_size: int = 64

    # train_split
    train_split: float = 0.1

    # epoch
    epoch: int = 200

    # early stopping patience
    early_stopping_patience: int = 3

    # model checkpoint path
    model_folder: str = "outputs"

    # model path
    model_ckpt_path: str = "outputs/model.pth"

    # data checkpoint path
    data_folder: str = "outputs"

    # device
    use_cuda: bool = False


def split_dataset(X: np.ndarray, Y: np.ndarray):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.1, random_state=42
    )
    return X_train, X_test, y_train, y_test


def pretrain_deepvelo_model(config: DeepveloConfig):
    adata, adata_raw = load_raw_dataset(
        config.raw_dataset_name,
        config.raw_dataset_path,
        read_cache=config.raw_dataset_use_cache,
    )
    X, Y = load_deepvelo_dataset(
        config.raw_dataset_path,
        config.deepvelo_dataset_path,
        read_cache=config.deepvelo_dataset_use_cache,
    )
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    # intialize model
    model_config = AEModelConfig(X.shape[-1], Y.shape[-1])
    early_stop_callback = EarlyStopping(
        patience=config.early_stopping_patience, threshold_mode="abs"
    )
    valid_ds = Dataset(X_test, Y_test)
    model = NeuralNet(
        module=ShallowAEModel,
        module__config=model_config,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=config.learning_rate,
        train_split=predefined_split(valid_ds),
        batch_size=config.batch_size,
        max_epochs=config.epoch,
        device="cuda" if config.use_cuda else "cpu",
        callbacks=[early_stop_callback],
        iterator_train__shuffle=True,
    )

    torch.save(asdict(config), f"{config.model_ckpt_path}.config")
    model.fit(X_train, Y_train)
    torch_module = model.module_
    torch.save(torch_module.state_dict(), config.model_ckpt_path)
