export OUTPUT_PATH=outputs/experiment/perturb/bonemarrow/feature_ablation
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py bonemarrow-feature-ablation \
    --pert.perturbation-num 10 \
    --slurm.mode slurm
