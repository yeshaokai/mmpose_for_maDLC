export CUDA_VISIBLE_DEVICES=0,1,2

python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_horse_512x512.py  --gpu-ids 1  --cfg-options optimizer.lr=0.0001 data.samples_per_gpu=32 --work-dir horse_lr1e-4bs32

 