export OUTPUT_PATH=outputs/experiment/perturb/synthetic_data/subset_sampling
export WANDB_API_KEY=YOUR_WANDB_KEY

DATASET_NAME=orange_skin_additive
DATA_SIZE=10000
FEATURE_DIM=100
python perturb_synthetic_data.py orange-skin-additive-subset-sampling-sampler-subset \
    --pert.perturbation-num 3 \
    --pert.subset-sampling.lr 1e-3 \
    --pert.subset-sampling.tau 0.1 \
    --pert.subset-sampling.tau-start 2 \
    --pert.subset-sampling.tau-anneal-steps 3000 \
    --pert.subset-sampling.hard \
    --head-trainer.train-num-steps 10000 \
    --head-trainer.synthetic-dataset-name $DATASET_NAME \
    --head-trainer.synthetic-data-size $DATA_SIZE \
    --head-trainer.synthetic-data-feature-dim $FEATURE_DIM \
    --trainer.synthetic-dataset-name $DATASET_NAME \
    --trainer.synthetic-data-size $DATA_SIZE \
    --trainer.synthetic-data-feature-dim $FEATURE_DIM \
    --trainer.pert-num-steps 500 \
    --trainer.save-and-sample-every 20
