export CUDA_VISIBLE_DEVICES=1,2
python tools/train.py    configs/bottom_up/resnet/pup/res50_pup_512x512.py  --gpu-ids 0 --cfg-options   optimizer.lr=0.000001 data.samples_per_gpu=4 --work-dir imagenet2pup_bs4_lr1e-6 &

python tools/train.py    configs/bottom_up/resnet/pup/res50_pup_512x512.py  --gpu-ids 1 --cfg-options   optimizer.lr=0.000001 data.samples_per_gpu=16 --work-dir imagenet2pup_bs16_lr1e-6 &

#python tools/train.py    configs/bottom_up/resnet/pup/res50_pup_512x512.py  --gpu-ids 1 --cfg-options   optimizer.lr=0.000001 data.samples_per_gpu=32 --work-dir imagenet2pup_bs32_lr1e-6
