DATETIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FOLDER=outputs/experiment/pert/mnist/pert_subset_$DATETIME

export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb_mnist.py \
    --pert.model-type subset_sampling \
    --pert.perturbation-num 64 \
    --pert.subset-sampling.no-use-scheduler \
    --pert.subset-sampling.lr 1e-3 \
    --pert.subset-sampling.tau 0.1 \
    --pert.subset-sampling.tau-start 2 \
    --pert.subset-sampling.tau-anneal-steps 3000 \
    --pert.subset-sampling.tau-scheduler-name linear \
    --pert.subset-sampling.hard \
    --pert.subset-sampling.sampler-pert-type parameterized_subset_sampling \
    --pert.subset-sampling.sampler-type subset \
    --pert.subset-sampling.parameterized-embedding-dim 512 \
    --pert.subset-sampling.parameterized-hidden-dim 64 \
    --pert.subset-sampling.optimizer-name Adam \
    --trainer.data-folder datasets/mnist \
    --trainer.starting-num-type 8 \
    --trainer.pert-num-steps 100 \
    --trainer.pert-state-step -1 \
    --trainer.save-and-sample-every 20 \
    --trainer.train-batch-size 64 \
    --trainer.eval-batch-size 64 \
    --trainer.num-workers 2 \
    --trainer.output-folder $OUTPUT_FOLDER \
    --head-trainer.data-folder datasets/mnist \
    --head-trainer.output-folder $OUTPUT_FOLDER/head_model \
    --head-trainer.train-num-steps 2000 \
    --head-trainer.save-and-eval-every 500 \
    --head-trainer.num-type-list 8 3 \
    --head-trainer.tgt-num-type 3 \
    --slurm.mode slurm \
    --slurm.slurm-output-folder $OUTPUT_FOLDER/slurm \
    --slurm.cpus-per-task 4 \
    --slurm.node_list laniakea \
    --wandb.name pert_subset_mnist_$DATETIME \
    --wandb.notes "using train test splits"