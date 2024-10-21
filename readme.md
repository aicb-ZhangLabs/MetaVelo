<h1 align="center"><strong>MetaVelo</strong></h1>

Our code is implemented using PyTorch and all experiments are run through the cluster job scheduler `slurm`. All jobs are submitted to clusters using the `submitit` package. To run in local, please set `slurm.mode` to `local`.

Our code is fully configurable. For any execution python scripts, use `-h` to see all arguments and descriptions. For example, `python train.py -h`. We use `wandb` to visualize the training process.

## File Structures

- `configs`: contains all experiment related configurations
- `scripts`: contains all bash scripts to launch experiments
- `src`: contains all source code
  - `pert`: source code of perturbation methods
  - `trainer`: source code of perturbation training logits
- `tests`: test code

## Dependencies

1. Python: 3.10.12
2. Install `scvelo` using the latest version at the master branch

    ```bash
    pip install git+https://github.com/theislab/scvelo@d89ca6aecbe93256fbcdd8a521fdee2b9f2a673a
    ```

3. Install other required packages

    ```bash
    pip install -r requirements.txt
    ```

## Data Preprocessing

We provided preprocessing scripts of three datasets in  `scripts/preprocess`, `scripts/pretrain`, and `scripts/pretrajectory`.

1. Preprocess dataset

    ```bash
    sh scripts/preprocess/preprocess_pancreas.sh
    ```

2. Sampled trajectories
The pretraining process would run on Slurm.

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

All lauch scripts are listed in `scripts/perturbation`.

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
