export CUDA_VISIBLE_DEVICES=0

for animal in rodent badja atrwtiger stanfordextra dog_cat_sheep_horse_cow
do	    
    python demo/bottom_up_tsne_demo.py configs/bottom_up/resnet/modelzoo/res50_superanimal_512x512.py checkpoints/res50_superanimal_epoch_500.pth --img-root data/cocomagic/$animal/images/train --json-file data/cocomagic/$animal/annotations/train.json --tag $animal
done
