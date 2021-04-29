export CUDA_VISIBLE_DEVICES=1

for animal in badja atrwtiger rodent stanfordextra dog_cat_sheep_horse_cow dlctopviewmouse topviewouse horse
do
    python demo/bottom_up_tsne_demo.py configs/bottom_up/resnet/modelzoo/res50_superanimal_512x512.py data/modelzoo_checkpoints/4good_epoch200.pth  --img-root data/cocomagic/$animal/images/val --json-file data/cocomagic/$animal/annotations/val.json --tag $animal
done





