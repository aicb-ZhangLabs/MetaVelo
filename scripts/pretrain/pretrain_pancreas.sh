export CUDA_VISIBLE_DEVICES=$1

python pretrain.py --model.raw_dataset_name pancreas \
    --model.raw_dataset_path outputs/pretrain/pancreas/data/pancreas.h5ad \
    --model.raw_dataset_use_cache \
    --model.deepvelo_dataset_path outputs/pretrain/pancreas/data/deepvelo_dataset.pth \
    --model.deepvelo_dataset_use_cache \
    --model.model_ckpt_path outputs/pretrain/pancreas/model/autoencoder.pth \
    --model.model_folder outputs/pretrain/pancreas/model \
    --model.data_folder outputs/pretrain/pancreas/data \
    --model.use_cuda
