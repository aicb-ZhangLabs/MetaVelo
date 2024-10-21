DATETIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FOLDER=outputs/experiment/train/pancreas/ode_model_all_$DATETIME

export WANDB_API_KEY=YOUR_WANDB_KEY
python train.py --model.dim 64 \
    --model.input-dim 668 \
    --trainer.train-batch-size 64 \
    --trainer.train-num-steps 2000 \
    --trainer.train-lr 1e-4 \
    --trainer.save-and-sample-every 500 \
    --trainer.num-workers 4 \
    --trainer.dataset-name pancreas \
    --trainer.data-folder datasets/pretrain/pancreas/trajs/sampled_traj.npz \
    --trainer.output-folder $OUTPUT_FOLDER \
    --trainer.ann_prc_data datasets/pretrain/pancreas/data/pancreas.h5ad \
    --trainer.ann_raw_data datasets/pretrain/pancreas/data/pancreas.h5ad.raw \
    --trainer.starting-cell-type Pre-endocrine \
    --trainer.train-only-starting-cell-type-data \
    --trainer.cell-type-ratio-keys Alpha Beta \
    --trainer.deep-velo-model-checkpoint outputs/pretrain/pancreas/model/autoencoder.pth \
    --trainer.deep-velo-model-dim 668 \
    --trainer.deep-velo-scaling-factor 0.8170944342283816 \
    --trainer.deep-velo-intermediate-step 3 \
    --trainer.t-span 0 1 \
    --trainer.t-step 65 \
    --trainer.seq-step 5 \
    --slurm.mode slurm \
    --slurm.slurm-output-folder $OUTPUT_FOLDER/slurm \
    --slurm.cpus-per-task 6 \
    --slurm.node_list laniakea \
    --wandb.name train_pancreas_ode_all_$DATETIME