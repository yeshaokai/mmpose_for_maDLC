export CUDA_VISIBLE_DEVICES=0
python demo/bottom_up_img_demo.py configs/bottom_up/resnet/3mouse/res50_3mouse_512x512.py checkpoints/res50_3mouse_512x512_epoch700_dlc.pth  --img-root data/3mouse/images --json-file data/3mouse/annotations/dlc_shuffle0_val.json --out-img-root output_images --kpt-thr 0.0 && cp output_images/*.png data/

#python demo/bottom_up_img_demo.py configs/bottom_up/hrnet/3mouse/hrnet_w32_3mouse_512x512.py checkpoints/hrnet_w32_3mouse_512x512_epoch800_dlc.pth  --img-root data/3mouse/images --json-file data/3mouse/annotations/dlc_shuffle0_val.json --out-img-root output_images --kpt-thr 0.1 && cp output_images/*.png data/ 


#python demo/bottom_up_img_demo.py configs/bottom_up/resnet/marmoset/res50_marmoset_512x512.py checkpoints/res50_marmoset_epoch_1000.pth  --img-root data/marmoset/images --json-file data/marmoset/annotations/dlc_shuffle0_val.json --out-img-root output_images --kpt-thr 0.1 && cp output_images/*.png data/


#python demo/bottom_up_img_demo.py configs/bottom_up/hrnet/marmoset/hrnet_w32_marmoset_512x512.py  checkpoints/hrnet_w32_marmoset_epoch_600.pth  --img-root data/marmoset/images --json-file data/marmoset/annotations/dlc_shuffle0_val.json --out-img-root output_images --kpt-thr 0.1 && cp output_images/*.png data/


