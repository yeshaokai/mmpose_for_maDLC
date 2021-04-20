export CUDA_VISIBLE_DEVICES=1,2


python tools/train.py    configs/bottom_up/resnet/pup/res50_pup_512x512.py  --gpu-ids 1 --cfg-options   optimizer.lr=0.001 data.samples_per_gpu=4 


