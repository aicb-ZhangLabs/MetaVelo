export OUTPUT_PATH=outputs/experiment/perturb/dentategyrus/permutation
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py dentategyrus-permutation-inverse-loss \
    --pert.perturbation-num 10 \
    --slurm.mode slurm
