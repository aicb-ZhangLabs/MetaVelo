<h1 align="center"><strong>MetaVelo</strong></h1>


<p align="center">
  <img src="assets/overview.jpg" alt="Overview of MetaVelo">
</p>

Code for the paper "Identifying Combinatorial Regulatory Genes for Cell Fate Decision via Reparameterizable Subset Explanations" (KDD 2025).

Our code is implemented using PyTorch and all experiments are run through the cluster job scheduler `slurm`. All jobs are submitted to clusters using the `submitit` package. To run locally, please set `slurm.mode` to `local`.

Our code is fully configurable. For any execution Python scripts, use `-h` to see all arguments and descriptions. For example, `python train.py -h`. We use `wandb` to visualize the training process.

If you use this code, please consider citing the paper using the BibTeX reference below.

```bibtex
@inproceedings{metavelo2025,
    title={Identifying Combinatorial Regulatory Genes for Cell Fate Decision via Reparameterizable Subset Explanations},
    author={Junhao Liu, Pengpeng Zhang, Martin Renqiang Min, and Jing Zhang},
    booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25), August 3--7, 2025, Toronto, ON, Canada},
    year={2025},
}
```

## File Structures

- `configs`: contains all experiment-related configurations
- `scripts`: contains all bash scripts to launch experiments
- `src`: contains all source code
  - `pert`: source code of perturbation methods
  - `trainer`: source code of perturbation training logic
- `tests`: test code

## Dependencies

- Python: 3.10.12
- PyTorch: 2.0.1
- Install `scvelo` using the latest version at the master branch

    ```bash
    pip install git+https://github.com/theislab/scvelo@d89ca6aecbe93256fbcdd8a521fdee2b9f2a673a
    ```

- Install other required packages

    ```bash
    pip install -r requirements.txt
    ```

## Data Preprocessing

We provide preprocessing scripts for three datasets in `scripts/preprocess`, `scripts/pretrain`, and `scripts/pretrajectory`.

1. Preprocess dataset

    ```bash
    sh scripts/preprocess/preprocess_pancreas.sh
    ```

2. Sample trajectories
The pretraining process will run on Slurm.

    ```bash
    sh scripts/pretrain/pretrain_pancreas.sh
    sh scripts/pretrajectory/pretrajectory_pancreas.sh
    ```

## Surrogate Model Training

To see argument details, use `python train.py -h`. This process would run on Slurm. All three launch scripts for each dataset are listed in `scripts/train`.

```bash
sh scripts/train/train_pancreas.sh
```

## Reparametrizable Subset Sampling for Perturbation

All launch scripts are listed in `scripts/perturbation`.

Run perturbation on the pretrained surrogate model. This process would run on Slurm. To see argument details, use `python perturb.py -h`. To reproduce our experiments,

```bash
sh scripts/perturbation/pancreas/pert_subset_sampling_new_subset_sampler.sh
```

To perturb on MNIST,

```bash
sh scripts/perturbation/mnist/pert_subset_sampling_subset_sampler.sh
```

## Acknowledgements

If you have any questions, feel free to reach out to us via email at `junhaoliu17@gmail.com`.
