export OUTPUT_PATH=outputs/experiment/perturb/pancreas/lime
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py pancreas-lime-inverse-loss \
    --pert.perturbation-num 10 \
    --pert.lime.mask-type random \
    --pert.lime.masked-batch-size 1024 \
    --pert.lime.neighbor-size 32768 \
    --pert.lime.neighbor-feat-num 120000 \
    --pert.lime.feature-selection auto \
    --pert.lime.distance-metric cosine \
    --trainer.pert-num-steps 100 \
    --trainer.save-and-sample-every 20 \
    --slurm.mode slurm
