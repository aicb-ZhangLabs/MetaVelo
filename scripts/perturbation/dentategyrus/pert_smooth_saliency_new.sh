export OUTPUT_PATH=outputs/experiment/perturb/dentategyrus/saliency
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py dentategyrus-smooth-saliency-inverse-loss \
    --pert.perturbation-num 10 \
    --pert.saliency.smooth-number 256 \
    --slurm.mode slurm
