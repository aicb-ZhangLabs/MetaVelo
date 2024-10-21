export OUTPUT_PATH=outputs/experiment/perturb/bonemarrow/mean_importance
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py bonemarrow-mean-importance \
    --pert.perturbation-num 10 \
    --slurm.mode slurm
