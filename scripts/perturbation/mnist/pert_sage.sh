DATETIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FOLDER=outputs/experiment/pert/mnist/pert_sage_$DATETIME

export WANDB_API_KEY=YOUR_WANDB_KEY
python perturb_mnist.py \
    --trainer-method sage \
    --pert.model-type sage \
    --pert.perturbation-num 10 \
    --pert.sage.trigger-pert-type trigger_perturbation \
    --pert.sage.n-jobs 1 \
    --pert.sage.batch-size 512 \
    --pert.sage.random-state 2024 \
    --pert.sage.n-permutations 2560000 \
    --pert.sage.no-detect-convergence \
    --pert.perturbation-num 32 \
    --trainer.data-folder datasets/mnist \
    --trainer.starting-num-type 8 \
    --trainer.pert-num-steps 500 \
    --trainer.pert-state-step -1 \
    --trainer.save-and-sample-every 10 \
    --trainer.train-batch-size 64 \
    --trainer.eval-batch-size 64 \
    --trainer.num-workers 2 \
    --trainer.output-folder $OUTPUT_FOLDER \
    --head-trainer.data-folder datasets/mnist \
    --head-trainer.output-folder $OUTPUT_FOLDER/head_model \
    --head-trainer.train-num-steps 2000 \
    --head-trainer.save-and-eval-every 500 \
    --head-trainer.num-type-list 8 3 \
    --head-trainer.tgt-num-type 8 \
    --slurm.mode debug \
    --slurm.slurm-output-folder $OUTPUT_FOLDER/slurm \
    --slurm.cpus-per-task 4 \
    --slurm.node_list laniakea \
    --wandb.project aurora \
    --wandb.name pert_sage_mnist_$DATETIME \
    --wandb.notes "using train test splits"