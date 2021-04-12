export CUDA_VISIBLE_DEVICES=1,2

#python tools/train.py    configs/bottom_up/resnet/fish/res50_fish_512x512.py  --gpu-ids 0  --cfg-options   optimizer.lr=0.00015 data.samples_per_gpu=8 --work-dir imagenet2fish_bs8_lr1e-4 &
#python tools/train.py    configs/bottom_up/resnet/fish/res50_fish_512x512.py  --gpu-ids 0  --cfg-options   optimizer.lr=0.00015 data.samples_per_gpu=16 --work-dir imagenet2fish_bs16_lr1e-4 &
#python tools/train.py    configs/bottom_up/resnet/fish/res50_fish_512x512.py  --gpu-ids 0  --cfg-options   optimizer.lr=0.00015 data.samples_per_gpu=32 --work-dir imagenet2fish_bs32_lr1e-4 

python tools/train.py configs/bottom_up/hrnet/fish/hrnet_w32_fish_512x512.py --gpu-ids 0
