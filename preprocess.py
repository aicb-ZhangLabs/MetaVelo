import os
import tyro
import scvelo as scv
import anndata as ad
import numpy as np

from dataclasses import dataclass


@dataclass
class PreprocessArgs:
    # raw dataset name
    raw_dataset_name: str = None

    # raw dataset to save path
    raw_dataset_path: str = None

    # data cache path for scvelo
    data_folder: str = "datasets"

    n_top_genes: int = 1000


def raw_dataset_processing(
    adata: ad.AnnData, adata_raw: ad.AnnData, args: PreprocessArgs
):
    scv.pp.filter_and_normalize(
        adata, min_shared_counts=20, n_top_genes=args.n_top_genes
    )
    scv.pp.filter_and_normalize(
        adata_raw, log=False, min_shared_counts=20, n_top_genes=args.n_top_genes
    )
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # extract dynamical velocity vectors
    scv.tl.recover_dynamics(adata, n_jobs=16)
    scv.tl.velocity(adata, mode="dynamical", use_raw=True)
    scv.tl.velocity_graph(adata, n_jobs=16)

    # special fixes for doing umap on different datasets
    if args.raw_dataset_name in {"dentategyrus", "forebrain"}:
        scv.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        scv.tl.umap(adata)
    scv.pl.velocity_embedding_stream(adata, basis="umap")

    # latent time inference
    scv.tl.latent_time(adata)
    scv.pl.scatter(adata, color="latent_time", color_map="gnuplot", size=80)

    # speical post fixes for standardizing different datasets
    if args.raw_dataset_name == "forebrain":
        adata.obs.rename(columns={"Clusters": "clusters"}, inplace=True)
        adata_raw.obs.rename(columns={"Clusters": "clusters"}, inplace=True)
        adata.uns["clusters_colors"] = np.array(
            [
                "#8fbc8f",
                "#f4a460",
                "#fdbf6f",
                "#ff7f00",
                "#b2df8a",
                "#1f78b4",
                "#6a3d9a",
            ],
            dtype=object,
        )
    return adata, adata_raw


def process_raw_dataset(args: PreprocessArgs):
    adata: ad.AnnData
    adata_raw: ad.AnnData

    dataset_name = args.raw_dataset_name
    cache_folder = args.data_folder
    dataset_path = args.raw_dataset_path

    cache_path = f"{cache_folder}/{dataset_name}/{dataset_name}.h5ad"
    if dataset_name == "pancreas":
        adata = scv.datasets.pancreas(cache_path)
    elif dataset_name == "dentategyrus":
        adata = scv.datasets.dentategyrus_lamanno()
        adata.write(cache_path)
    elif dataset_name == "forebrain":
        adata = scv.datasets.forebrain()
        adata.write(cache_path)
    else:
        raise Exception(f"Cannot found dataset {dataset_name}")
    adata_raw = adata.copy()

    # preprocess dataset
    adata, adata_raw = raw_dataset_processing(adata, adata_raw, args)
    adata.write(dataset_path)
    adata_raw.write(f"{dataset_path}.raw")
    return adata, adata_raw


if __name__ == "__main__":
    args = tyro.cli(PreprocessArgs)
    os.makedirs(args.data_folder, exist_ok=True)
    process_raw_dataset(args)
