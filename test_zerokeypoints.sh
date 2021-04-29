export CUDA_VISIBLE_DEVICES=0,1,2

for train_num in 16 
do
    python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_rodent_512x512.py  --gpu-ids 1 --cfg-options    data.train.ann_file=data/cocomagic/rodent_all/annotations/train_$train_num.json  optimizer.lr=0.00015 data.samples_per_gpu=12   --work-dir temp_train$train_num
done 
    


