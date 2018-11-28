# CUDA_VISIBLE_DEVICES=0 python train.py --dataset XM2VTS --nEpochs 200 --cuda
# CUDA_VISIBLE_DEVICES=0 python test.py --dataset XM2VTS --p_model photo_G_1_model_epoch_200.pth --s_model sketch_G_2_model_epoch_200.pth --cuda

# CUDA_VISIBLE_DEVICES=1 python train.py --dataset AR --nEpochs 200 --cuda
# CUDA_VISIBLE_DEVICES=1 python test.py --dataset AR --p_model photo_G_1_model_epoch_200.pth --s_model sketch_G_2_model_epoch_200.pth --cuda

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CUHKStudent --nEpochs 200 --cuda
# CUDA_VISIBLE_DEVICES=0 python test.py --dataset CUHKStudent --p_model photo_G_1_model_epoch_200.pth --s_model sketch_G_2_model_epoch_200.pth --cuda

CUDA_VISIBLE_DEVICES=2 python train.py --dataset CUHKFERET --nEpochs 200 --cuda
CUDA_VISIBLE_DEVICES=2 python test.py --dataset CUHKFERET --p_model photo_G_1_model_epoch_200.pth --s_model sketch_G_2_model_epoch_200.pth --cuda
