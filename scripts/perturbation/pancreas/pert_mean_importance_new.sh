export OUTPUT_PATH=outputs/experiment/perturb/pancreas/mean_importance
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py pancreas-mean-importance \
    --pert.perturbation-num 10 \
    --slurm.mode slurm
