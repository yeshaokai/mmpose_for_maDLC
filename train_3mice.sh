export CUDA_VISIBLE_DEVICES=1,2

python tools/train.py    configs/bottom_up/resnet/3mouse/res50_3mouse_512x512.py  --gpu-ids 0
#bash tools/dist_train.sh    configs/bottom_up/hrnet/3mouse/hrnet_w32_3mouse_512x512.py 
