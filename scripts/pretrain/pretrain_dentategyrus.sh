python pretrain.py --model.raw_dataset_name dentategyrus \
    --model.raw_dataset_path datasets/pretrain/dentategyrus/data/dentategyrus.h5ad \
    --model.raw_dataset_use_cache \
    --model.deepvelo_dataset_path datasets/pretrain/dentategyrus/data/dentategyrus.pth \
    --model.deepvelo_dataset_use_cache \
    --model.model_ckpt_path datasets/pretrain/dentategyrus/model/autoencoder.pth \
    --model.model_folder datasets/pretrain/dentategyrus/model \
    --model.data_folder datasets/pretrain/dentategyrus/data \
    --model.use_cuda