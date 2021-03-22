export CUDA_VISIBLE_DEVICES=0
python tools/train.py    configs/bottom_up/resnet/modelzoo/res50_5animal_finetune_on_rodent_512x512.py  --gpu-ids 0
#bash tools/dist_train.sh    configs/bottom_up/resnet/modelzoo/res50_5animal_finetune_on_rodent_512x512.py  

