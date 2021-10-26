export CUDA_VISIBLE_DEVICES=2

#python tools/test.py    configs/bottom_up/resnet/3mouse/res50_3mouse_512x512.py  work_dirs/res50_3mouse_512x512/best.pth --cfg-options data.test.ann_file=data/3mice/annotations/dlc_shuffle1_train.json

#python tools/test.py    configs/bottom_up/hrnet/3mouse/hrnet_w32_3mouse_512x512.py  work_dirs/hrnet_w32_3mouse_512x512/best.pth --cfg-options data.test.ann_file=data/3mice/annotations/dlc_shuffle1_train.json 

#python tools/train.py    configs/bottom_up/resnet/3mouse/res50_3mouse_512x512.py  
#python tools/train.py    configs/bottom_up/hrnet/3mouse/hrnet_w32_3mouse_512x512.py 
