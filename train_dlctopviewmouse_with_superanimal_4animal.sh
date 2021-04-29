export CUDA_VISIBLE_DEVICES=0,1,2

for train_num in 16 32 64 128 852
do
    python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_dlctopviewmouse_512x512.py  --gpu-ids 1 --cfg-options data.train.ann_file=data/cocomagic/dlctopviewmouse/annotations/train_$train_num.json   load_from=data/modelzoo_checkpoints/superanimal_4animal.pth  optimizier.lr=0.0015 data.samples_per_gpu=12     --work-dir superanima_4animal_l2dlctopviewmouse_train$train_num 
done || exit 1
    
 
