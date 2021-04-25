export CUDA_VISIBLE_DEVICES=0,1,2

for train_num in 3299
do
    python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_tigers_512x512.py  --gpu-ids 0 --cfg-options data.train.ann_file=data/cocomagic/atrwtiger/annotations/train_$train_num.json   load_from=data/modelzoo_checkpoints/superanimal_epoch200.pth  optimizer.lr=0.0015 data.samples_per_gpu=32   --work-dir superanimal2tigers_train$train_num  
done || exit 1
    
 
