import os
import tyro
import json
import torch
import scanpy as sc
import numpy as np
import sklearn

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, asdict
from sklearn.model_selection import train_test_split
from src.trajectory.interpolate import torchdyn_interpolate
from src.trajectory.sample import sample_deep_velo_trajectory, AESys
from src.model.ae import ShallowAEModel, AEModelConfig


@dataclass
class Args:
    ann_prc_data: str
    ann_raw_data: str
    npy_prc_data: str
    model_checkpoint: str
    traj_output_dir: str
    cuda: bool = False

    # sampled trajectory settings
    sampled_traj_file: str = ""

    # random seed
    random_state: int = 2023


def sample_trajectory(
    adata,
    adata_raw,
    X,
    autoencoder,
    scaling_factor,
    max_steps: int,
    intermediate_step: int,
    use_cuda: bool,
):
    # use PCA to reduce the dim of the count data
    pca = sklearn.decomposition.PCA(n_components=30)
    adata_pca = pca.fit_transform(
        np.log1p(adata_raw.X.A[:, adata.var["velocity_genes"]])
    )

    # construct KNN with PCA
    neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=30)
    neigh.fit(adata_pca)

    # with open("dataset/pancreas/models/tools.pkl", "rb") as output:
    #     data = pickle.load(output)
    #     pca = data[0]
    #     umap_reducer = data[1]
    #     neigh = data[2]

    # simulate trajectories for all cells in 30 time steps
    dataset = TensorDataset(
        torch.from_numpy(adata_raw.X[:, adata.var["velocity_genes"]].A)
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    intergrate_fn = AESys(autoencoder=autoencoder, scaling_factor=scaling_factor)

    # use cuda
    if use_cuda:
        intergrate_fn = intergrate_fn.cuda()

    all_paths = []
    all_steps = None
    for (in_x,) in tqdm(dataloader):
        path, time_steps = torchdyn_interpolate(
            in_x,
            X=adata_raw.X.A[:, adata.var["velocity_genes"]],
            int_fn=intergrate_fn,
            neigh=neigh,
            pca=pca,
            steps=max_steps,
            intermediate_steps=intermediate_step,
            use_cuda=use_cuda,
        )
        all_paths.append(path)
        all_steps = time_steps

    return np.concatenate(all_paths, axis=1), all_steps


def generate_trajectory(args: Args):
    ### read data
    adata = sc.read_h5ad(args.ann_prc_data)
    adata_raw = sc.read_h5ad(args.ann_raw_data)

    npz_dict = torch.load(args.npy_prc_data)
    X = npz_dict["X"]
    Y = npz_dict["Y"]

    ### load model
    model_dim = X.shape[-1]
    autoencoder = ShallowAEModel(AEModelConfig(model_dim, model_dim))
    autoencoder.load_state_dict(torch.load(args.model_checkpoint))

    ### heuristic settings
    # Heurstic scaling factor for the normalized magnitude when integrating
    scaling_factor = np.percentile(Y.mean(axis=0) / X.mean(axis=0), 99)

    # Heurstic maximum step size for the normalized magnitude when integrating
    total_steps = int(np.percentile(np.abs(X.std(axis=0) * 2 / Y.mean(axis=0)), 80))
    intermediate_step = 3
    max_steps = total_steps // intermediate_step

    # make output dir
    os.makedirs(args.traj_output_dir, exist_ok=True)

    meta = asdict(args)
    with open(f"{args.traj_output_dir}/meta.json", "w") as f:
        meta.update(
            {
                "scaling_factor": scaling_factor,
                "total_steps": total_steps,
                "intermediate_step": intermediate_step,
                "max_steps": max_steps,
                "model_dim": model_dim,
            }
        )
        json.dump(meta, f)

    sampled_path, sampled_time_step = sample_deep_velo_trajectory(
        adata,
        adata_raw,
        autoencoder=autoencoder,
        initial_state=torch.from_numpy(adata_raw.X[:, adata.var["velocity_genes"]].A),
        scaling_factor=scaling_factor,
        max_steps=max_steps,
        intermediate_step=intermediate_step,
        device=torch.device("cuda" if args.cuda else "cpu"),
        random_state=args.random_state,
    )

    traj_path = f"{args.traj_output_dir}/sampled_traj.npz"
    np.savez(traj_path, path=sampled_path, step=sampled_time_step)
    return traj_path


def split_subset(args: Args, traj_data: np.ndarray):
    adata = sc.read_h5ad(args.ann_prc_data)
    cell_types = np.array(adata.obs.clusters)

    ind = np.arange(traj_data["path"].shape[1])
    train_ind, test_ind = train_test_split(
        ind, test_size=0.2, random_state=args.random_state
    )
    train_ind, val_ind = train_test_split(
        train_ind, test_size=0.2, random_state=args.random_state
    )

    subsets = {
        "train": train_ind,
        "val": val_ind,
        "test": test_ind,
    }
    for subset_name, subset_ind in subsets.items():
        # show statistics
        cell_type, cell_count = np.unique(cell_types[subset_ind], return_counts=True)

        print()
        print(subset_name, "subset statistics:")
        for ct, cc in zip(cell_type, cell_count):
            print(f"{ct}: {cc} (cc / total: {cc / cell_count.sum():.2f})")

        # save subset
        subset_path = f"{args.traj_output_dir}/sampled_traj_{subset_name}.npz"
        np.savez(
            subset_path,
            path=traj_data["path"][:, subset_ind],
            step=traj_data["step"],
            global_ind=subset_ind,
            cell_type=cell_types[subset_ind],
        )


def main():
    args = tyro.cli(Args)

    traj_path = args.sampled_traj_file
    if not args.sampled_traj_file:
        traj_path = generate_trajectory(args)

    # read trajectory data
    traj_data = np.load(traj_path)
    split_subset(args, traj_data)


if __name__ == "__main__":
    main()
