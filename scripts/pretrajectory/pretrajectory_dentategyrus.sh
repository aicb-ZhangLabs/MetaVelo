export CUDA_VISIBLE_DEVICES=$1

python pretrajectory.py --ann_prc_data datasets/pretrain/dentategyrus/data/dentategyrus.h5ad \
    --ann_raw_data datasets/pretrain/dentategyrus/data/dentategyrus.h5ad.raw \
    --npy_prc_data datasets/pretrain/dentategyrus/data/dentategyrus.pth \
    --model_checkpoint datasets/pretrain/dentategyrus/model/autoencoder.pth \
    --traj_output_dir datasets/pretrain/dentategyrus/trajs \
    --cuda