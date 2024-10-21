export OUTPUT_PATH=outputs/experiment/perturb/pancreas/sage
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py pancreas-sage-inverse-loss \
    --pert.perturbation-num 10 \
    --pert.sage.n-permutations 256000\
    --trainer.pert-num-steps 100 \
    --trainer.save-and-sample-every 20 \
    --slurm.mode slurm
