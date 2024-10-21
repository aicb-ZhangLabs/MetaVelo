export OUTPUT_PATH=outputs/experiment/perturb/pancreas/cxplain
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py pancreas-cxplain-inverse-loss \
    --pert.perturbation-num 10 \
    --pert.cxplain.batch-size 128 \
    --pert.cxplain.trainer-config.batch-size 128 \
    --pert.cxplain.trainer-config.epoch 500 \
    --pert.cxplain.trainer-config.early-stopping-patience 10 \
    --pert.cxplain.trainer-config.learning-rate 5e-4 \
    --pert.cxplain.trainer-config.downsample-factors 2 \
    --pert.cxplain.trainer-config.num-units 668 \
    --trainer.pert-num-steps 100 \
    --trainer.save-and-sample-every 20 \
    --slurm.mode slurm
