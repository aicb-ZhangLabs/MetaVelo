export CUDA_VISIBLE_DEVICES=$1

python pretrajectory.py --ann_prc_data datasets/pretrain/forebrain/data/forebrain.h5ad \
    --ann_raw_data datasets/pretrain/forebrain/data/forebrain.h5ad.raw \
    --npy_prc_data datasets/pretrain/forebrain/data/forebrain.pth \
    --model_checkpoint datasets/pretrain/forebrain/model/autoencoder.pth \
    --traj_output_dir datasets/pretrain/forebrain/trajs \
    --cuda