export CUDA_VISIBLE_DEVICES=2

python tools/test.py    configs/bottom_up/resnet/pup/res50_pup_512x512.py  work_dirs/res50_pup_512x512/best.pth --cfg-options data.test.ann_file=data/pup/annotations/dlc_shuffle1_val.json

#python tools/test.py    configs/bottom_up/hrnet/pup/hrnet_w32_pup_512x512.py  work_dirs/hrnet_w32_pup_512x512/best.pth --cfg-options data.test.ann_file=data/pup/annotations/dlc_shuffle1_train.json 

#python tools/train.py    configs/bottom_up/resnet/pup/res50_pup_512x512.py  
#python tools/train.py    configs/bottom_up/hrnet/pup/hrnet_w32_pup_512x512.py


