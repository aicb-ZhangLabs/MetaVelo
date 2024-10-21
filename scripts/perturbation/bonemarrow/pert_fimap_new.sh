export OUTPUT_PATH=outputs/experiment/perturb/bonemarrow/fimap
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py bonemarrow-fimap \
    --pert.perturbation-num 10 \
    --pert.fimap.tau 0.5 \
    --pert.fimap.regularization-weight 1e-1 \
    --pert.fimap.optimizer-name Adam \
    --pert.fimap.lr 1e-3 \
    --trainer.pert-num-steps 500 \
    --trainer.save-and-sample-every 100 \
    --slurm.mode slurm
