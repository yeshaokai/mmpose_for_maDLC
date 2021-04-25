export CUDA_VISIBLE_DEVICES=1

#for animal in badja atrwtiger stanfordextra dog_cat_sheep_horse_cow
#do
#done
#animal=atrwtiger

#python demo/bottom_up_tsne_demo.py configs/bottom_up/resnet/modelzoo/res50_superanimal_512x512.py checkpoints/res50_superanimal_epoch_500.pth --img-root data/cocomagic/$animal/images/val --json-file data/cocomagic/$animal/annotations/val.json --tag $animal

python demo/bottom_up_tsne_demo.py configs/bottom_up/hrnet/pup/hrnet_w32_pup_512x512.py  work_dirs/hrnet_w32_pup_512x512/epoch_1100.pth  --img-root data/pup/images --json-file data/pup/annotations/dlc_shuffle0_val.json --out-img-root output_images --kpt-thr 0.0 && cp output_images/*.png data/
