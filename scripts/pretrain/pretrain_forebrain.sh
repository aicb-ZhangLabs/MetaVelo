python pretrain.py --model.raw_dataset_name forebrain \
    --model.raw_dataset_path datasets/pretrain/forebrain/data/forebrain.h5ad \
    --model.raw_dataset_use_cache \
    --model.deepvelo_dataset_path datasets/pretrain/forebrain/data/forebrain.pth \
    --model.model_ckpt_path datasets/pretrain/forebrain/model/autoencoder.pth \
    --model.model_folder datasets/pretrain/forebrain/model \
    --model.data_folder datasets/pretrain/forebrain/data \
    --model.use_cuda