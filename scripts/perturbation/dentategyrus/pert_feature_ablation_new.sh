export OUTPUT_PATH=outputs/experiment/perturb/dentategyrus/feature_ablation
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py dentategyrus-feature-ablation \
    --pert.perturbation-num 10 \
    --slurm.mode slurm
