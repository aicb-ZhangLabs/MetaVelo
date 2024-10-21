export OUTPUT_PATH=outputs/experiment/perturb/bonemarrow/sage
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py bonemarrow-sage-binary-head \
    --head-trainer.train-num-steps 1000 \
    --pert.perturbation-num 10 \
    --pert.sage.n-permutations 256000\
    --trainer.pert-num-steps 100 \
    --trainer.save-and-sample-every 20 \
    --slurm.mode slurm
