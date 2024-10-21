export OUTPUT_PATH=outputs/experiment/perturb/bonemarrow/approximation
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py bonemarrow-approximation-inverse-loss \
    --pert.perturbation-num 10 \
    --slurm.mode slurm
