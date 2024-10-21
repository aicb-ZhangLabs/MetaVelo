export OUTPUT_PATH=outputs/experiment/perturb/bonemarrow/subset_sampling
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py bonemarrow-subset-sampling \
    --head-trainer.train-num-steps 5000 \
    --pert.perturbation-num 10 \
    --pert.subset-sampling.lr 1e-3 \
    --pert.subset-sampling.tau 0.1 \
    --pert.subset-sampling.tau-start 2 \
    --pert.subset-sampling.tau-anneal-steps 3000 \
    --pert.subset-sampling.no-hard
