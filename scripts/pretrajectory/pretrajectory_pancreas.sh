export CUDA_VISIBLE_DEVICES=$1

python pretrajectory.py --ann_prc_data outputs/pretrain/pancreas/data/pancreas.h5ad \
    --ann_raw_data outputs/pretrain/pancreas/data/pancreas.h5ad.raw \
    --npy_prc_data outputs/pretrain/pancreas/data/deepvelo_dataset.pth \
    --model_checkpoint outputs/pretrain/pancreas/model/autoencoder.pth \
    --traj_output_dir outputs/pretrain/pancreas/trajs \
    --cuda