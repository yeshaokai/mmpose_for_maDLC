export CUDA_VISIBLE_DEVICES=0

#python tools/test.py  configs/bottom_up/resnet/3mouse/res50_3mouse_512x512.py work_dirs/res50_3mouse_512x512/epoch_700.pth
#python tools/test.py   configs/bottom_up/hrnet/3mouse/hrnet_w32_3mouse_512x512.py checkpoints/hrnet_w32_3mouse_512x512_epoch800_dlc.pth

#python tools/test.py  configs/bottom_up/resnet/marmoset/res50_marmoset_512x512.py checkpoints/res50_marmoset_epoch_1000.pth  

#python tools/test.py  configs/bottom_up/hrnet/marmoset/hrnet_w32_marmoset_512x512.py checkpoints/hrnet_w32_marmoset_epoch_600.pth

python tools/test.py configs/bottom_up/resnet/modelzoo/res50_rodent_512x512.py checkpoints/res50_dogs_epoch150_imagenet.pth
