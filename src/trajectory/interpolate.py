import torch
import numpy as np

from torchdyn.numerics import odeint
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from .utils import robust_log1p


def fast_interpolate(
    y0: np.ndarray,
    X: np.ndarray,
    int_fn,
    neigh: NearestNeighbors,
    pca: PCA,
    steps: int,
    intermediate_steps: int,
):
    """
    Faster version of the previous function, using RK23 instead of DOP853
    """
    solution = []
    for _ in range(steps):
        # Interpolate using autoencoder
        t_eval = list(range(intermediate_steps))
        sol = solve_ivp(int_fn, [0, max(t_eval)], y0, method="RK23", t_eval=t_eval)

        # only use the last step
        y = sol.y.T[-1:]

        # lower dimensionality
        ending_pt_pca = pca.transform(np.nan_to_num(np.log1p(y)))

        # find knn reference points
        _, interp_neigh = neigh.kneighbors(ending_pt_pca)

        y0 = np.mean(X[interp_neigh[0]], axis=0)
        solution.append(y0)

    return np.array(solution)


@torch.inference_mode()
def torchdyn_interpolate(
    y0: torch.Tensor,
    X: np.ndarray,
    int_fn,
    neigh: NearestNeighbors,
    pca: PCA,
    steps: int,
    intermediate_steps: int,
    device: torch.DeviceObjType,
):
    """
    Batch intergration through torchdyn
    """
    solution = [y0.numpy(force=True)]
    time_steps = [0]

    # Interpolate using autoencoder
    t_eval = torch.linspace(0, intermediate_steps - 1, intermediate_steps)

    for step in range(1, steps + 1):
        # get tensor
        y0_pth = torch.from_numpy(y0) if isinstance(y0, np.ndarray) else y0
        y0_pth = y0_pth.to(device)

        _, sol = odeint(int_fn, y0_pth, t_span=t_eval, solver="rk4")  # sol: [t, b, h]

        # only use the last step
        y = sol[-1].numpy(force=True)

        # lower dimensionality
        # ending_pt_pca = pca.transform(np.nan_to_num(np.log1p(y)))
        ending_pt_pca = pca.transform(robust_log1p(y))

        # find knn reference points
        _, interp_neigh = neigh.kneighbors(ending_pt_pca)

        # aggregate results
        batch_num = interp_neigh.shape[0]
        neigh_num = interp_neigh.shape[-1]
        neigh_points = X[interp_neigh.flatten()].reshape(batch_num, neigh_num, -1)
        y0 = np.mean(neigh_points, axis=1)

        solution.append(y0)
        time_steps.append(step)

    return np.array(solution), np.array(time_steps)
