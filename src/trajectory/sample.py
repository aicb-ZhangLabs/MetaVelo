from typing import Union
import torch
import torch.nn as nn
import sklearn
import numpy as np

from anndata import AnnData
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset
from .interpolate import torchdyn_interpolate
from ..pert.base import PertModule
from ..model.ae import ShallowAEModel, AEModelConfig


class AESys(nn.Module):
    def __init__(self, autoencoder: nn.Module, scaling_factor: float):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.model = autoencoder

    def forward(self, t, x, **kwargs):
        dx = self.model(x) * self.scaling_factor
        return dx


@torch.inference_mode()
def sample_deep_velo_trajectory(
    adata: AnnData,
    adata_raw: AnnData,
    autoencoder: nn.Module,
    initial_state: Union[torch.Tensor, DataLoader],
    scaling_factor: float,
    max_steps: int,
    intermediate_step: int,
    device: torch.DeviceObjType,
    random_state: int,
    pca_n_components: int = 30,
    knn_n_neighbors: int = 30,
    batch_size: int = 128,
    pre_log1p: bool = False,
):
    """Sampling trajectory via deep velo model

    :param adata: _description_
    :param adata_raw: _description_
    :param autoencoder: _description_
    :param initial_state: _description_
    :param scaling_factor: _description_
    :param max_steps: _description_
    :param intermediate_step: _description_
    :param device: _description_
    :param random_state: _description_
    :param pca_n_components: _description_, defaults to 30
    :param knn_n_neighbors: _description_, defaults to 30
    :param batch_size: _description_, defaults to 128
    :param pert_model: _description_, defaults to None
    :param pre_log1p: if intial_state already take log1p , defaults to False
    :return: sampled trajectory, time steps
    """
    # use PCA to reduce the dim of the count data
    pca = sklearn.decomposition.PCA(
        n_components=pca_n_components, random_state=random_state
    )
    adata_pca = pca.fit_transform(
        np.log1p(adata_raw.X.A[:, adata.var["velocity_genes"]])
    )

    # construct KNN with PCA
    neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=knn_n_neighbors)
    neigh.fit(adata_pca)

    # simulate trajectories for all cells from initial state
    if isinstance(initial_state, torch.Tensor):
        dataset = TensorDataset(initial_state)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        dataloader = initial_state

    # move to device
    intergrate_fn = AESys(autoencoder=autoencoder, scaling_factor=scaling_factor)
    intergrate_fn = intergrate_fn.to(device)

    all_paths = []
    all_steps = None
    for in_x in dataloader:
        if not isinstance(in_x, torch.Tensor):
            in_x = in_x[0]
        in_x = in_x.to(device)

        if pre_log1p:
            # cast to non log1p
            in_x = torch.expm1(in_x)
        path, time_steps = torchdyn_interpolate(
            in_x,
            X=adata_raw.X.A[:, adata.var["velocity_genes"]],
            int_fn=intergrate_fn,
            neigh=neigh,
            pca=pca,
            steps=max_steps,
            intermediate_steps=intermediate_step,
            device=device,
        )
        all_paths.append(path)
        all_steps = time_steps

    return np.concatenate(all_paths, axis=1), all_steps


@torch.inference_mode()
def sample_deep_velo_trajectory_from_ckpt(
    adata: AnnData,
    adata_raw: AnnData,
    model_ckpt: str,
    model_input_dim: int,
    initial_state: Union[torch.Tensor, DataLoader],
    *args,
    **kwargs
):
    autoencoder = ShallowAEModel(AEModelConfig(model_input_dim, model_input_dim))
    autoencoder.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
    autoencoder.eval()
    return sample_deep_velo_trajectory(
        adata, adata_raw, autoencoder, initial_state, *args, **kwargs
    )


@torch.inference_mode()
def sample_trajectory(
    model: nn.Module,
    t_span: torch.Tensor,
    seq_step: int,
    t_offset: int,
    dataloader: DataLoader,
    device: torch.DeviceObjType,
    pert_model: PertModule = None,
    pert_model_kwargs: dict = {},
):
    sampled_trajs = []
    ground_trajs = []
    for data in dataloader:
        data = rearrange(data, "b t h -> t b h")[:seq_step].to(device)
        inp = data[0]

        # add perturbation if needed
        if pert_model is not None:
            inp = pert_model.perturb(inp, **pert_model_kwargs)

        _, pred_traj = model(inp, t_span)
        sampled_trajs.append(pred_traj[::t_offset])
        ground_trajs.append(data)

    all_trajs = torch.cat(sampled_trajs, dim=1)  # [t, b, h]
    grd_trajs = torch.cat(ground_trajs, dim=1)  # [t, b, h]
    return grd_trajs, all_trajs
