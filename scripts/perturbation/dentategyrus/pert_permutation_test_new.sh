export OUTPUT_PATH=outputs/experiment/perturb/dentategyrus/permutation_test
export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb.py dentategyrus-permutation-test \
    --pert.perturbation-num 10 \
    --slurm.mode slurm
