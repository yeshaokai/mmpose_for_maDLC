export CUDA_VISIBLE_DEVICES=0,1,2


python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_rodent_512x512.py  --gpu-ids 0 --cfg-options data.train.ann_file=data/modelzoo/rodent/annotations/train_64.json  --work-dir super2rodents_train64 &
 
python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_rodent_512x512.py  --gpu-ids 1 --cfg-options data.train.ann_file=data/modelzoo/rodent/annotations/train_32.json  --work-dir super2rodents_train32 &

python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_rodent_512x512.py  --gpu-ids 1 --cfg-options data.train.ann_file=data/modelzoo/rodent/annotations/train.json  --work-dir super2rodents_train254 &
 
python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_rodent_512x512.py  --gpu-ids 2 --cfg-options data.train.ann_file=data/modelzoo/rodent/annotations/train_128.json  --work-dir super2rodents_train128 &


python -u tools/train.py  configs/bottom_up/resnet/modelzoo/res50_rodent_512x512.py  --gpu-ids 2 --cfg-options data.train.ann_file=data/modelzoo/rodent/annotations/train_16.json  --work-dir super2rodents_train16




