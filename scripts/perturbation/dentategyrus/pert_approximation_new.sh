export OUTPUT_PATH=outputs/experiment/perturb/dentategyrus/approximation
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py dentategyrus-approximation-inverse-loss \
    --pert.perturbation-num 10 \
    --slurm.mode slurm
