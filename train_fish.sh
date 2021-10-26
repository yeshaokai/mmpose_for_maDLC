export CUDA_VISIBLE_DEVICES=2


#python tools/test.py    configs/bottom_up/resnet/fish/res50_fish_512x512.py  work_dirs/res50_fish_512x512/best.pth --cfg-options data.test.ann_file=data/fish/annotations/dlc_shuffle1_train.json

python tools/test.py    configs/bottom_up/hrnet/fish/hrnet_w32_fish_512x512.py  work_dirs/hrnet_w32_fish_512x512/best.pth --cfg-options data.test.ann_file=data/fish/annotations/dlc_shuffle1_train.json 

#python tools/train.py    configs/bottom_up/resnet/fish/res50_fish_512x512.py  
#python tools/train.py configs/bottom_up/hrnet/fish/hrnet_w32_fish_512x512.py 
