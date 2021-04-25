export CUDA_VISIBLE_DEVICES=0

python tools/test.py configs/bottom_up/resnet/modelzoo/res50_rodent_512x512.py data/modelzoo_checkpoints/topviewmouse.pth --cfg-options data.test.ann_file=data/cocomagic/topviewmouse/annotations/train.json data.test.img_prefix=data/cocomagic/topviewmouse/images/train data.val.ann_file=data/cocomagic/topviewmouse/annotations/train.json data.val.img_prefix=data/cocomagic/topviewmouse/images/train
