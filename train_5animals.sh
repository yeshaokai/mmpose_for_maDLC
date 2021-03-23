export CUDA_VISIBLE_DEVICES=1,2
#python tools/train.py    configs/bottom_up/resnet/modelzoo/res50_dog_cat_sheep_horse_cow_512x512.py  --gpus 2
bash tools/dist_train.sh    configs/bottom_up/resnet/modelzoo/res50_dog_cat_sheep_horse_cow_512x512.py  

