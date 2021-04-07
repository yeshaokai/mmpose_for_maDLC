export CUDA_VISIBLE_DEVICES=0,1,2

python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_tigers_512x512.py --gpu-ids 0  &
python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_badja_512x512.py --gpu-ids 1
