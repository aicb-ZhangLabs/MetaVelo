export OUTPUT_PATH=outputs/experiment/perturb/bonemarrow/permutation
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py bonemarrow-permutation-inverse-loss \
    --pert.perturbation-num 10 \
    --slurm.mode slurm
