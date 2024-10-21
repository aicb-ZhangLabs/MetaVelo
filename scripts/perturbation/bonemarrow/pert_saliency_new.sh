export OUTPUT_PATH=outputs/experiment/perturb/bonemarrow/saliency
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py bonemarrow-saliency-inverse-loss \
    --pert.perturbation-num 10 \
    --slurm.mode slurm
