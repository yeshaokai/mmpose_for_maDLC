export CUDA_VISIBLE_DEVICES=0


python demo/bottom_up_tsne_demo.py configs/bottom_up/resnet/modelzoo/res50_superanimal_512x512.py checkpoints/res50_superanimal_epoch_500.pth --img-root data/modelzoo/rodent/images/val --json-file data/modelzoo/rodent/annotations/val.json

