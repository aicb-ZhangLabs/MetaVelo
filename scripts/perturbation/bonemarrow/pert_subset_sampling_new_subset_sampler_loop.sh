export OUTPUT_PATH=outputs/experiment/perturb/bonemarrow/subset_sampling
export WANDB_API_KEY=YOUR_WANDB_KEY

for i in {1..4}
do
    python perturb.py bonemarrow-subset-sampling-sampler-subset \
        --head-trainer.train-num-steps 5000 \
        --pert.perturbation-num $i \
        --pert.subset-sampling.lr 1e-3 \
        --pert.subset-sampling.tau 0.1 \
        --pert.subset-sampling.tau-start 2 \
        --pert.subset-sampling.tau-anneal-steps 3000 \
        --pert.subset-sampling.hard \
        --trainer.pert-num-steps 100 \
        --trainer.save-and-sample-every 20
done