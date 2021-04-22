export CUDA_VISIBLE_DEVICES=0,1,2

python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_topviewmouse_512x512.py  --gpu-ids 2  --cfg-options optimizer.lr=0.00015 data.samples_per_gpu=4 --work-dir topviewmouse_lr1e-4bs4 

 
