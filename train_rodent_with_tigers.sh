export CUDA_VISIBLE_DEVICES=0,1,2

for train_num in 16 32 64 128 357
do
    python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_rodent_512x512.py  --gpu-ids 1 --cfg-options data.train.ann_file=data/cocomagic/rodent_all/annotations/train_$train_num.json   load_from=data/modelzoo_checkpoints/tiger.pth    --work-dir tiger2rodents_train$train_num 
done || exit 1
    
 
