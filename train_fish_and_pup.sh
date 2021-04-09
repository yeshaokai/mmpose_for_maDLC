export CUDA_VISIBLE_DEVICES=1,2

python tools/train.py    configs/bottom_up/resnet/fish/res50_fish_512x512.py  --gpu-ids 0   &

python tools/train.py    configs/bottom_up/resnet/pup/res50_pup_512x512.py  --gpu-ids 1

