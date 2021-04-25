export CUDA_VISIBLE_DEVICES=0
#python demo/bottom_up_tsne_demo.py configs/bottom_up/hrnet/3mouse/hrnet_w32_3mouse_512x512.py  checkpoints/hrnet_w32_3mouse_512x512_epoch800_dlc.pth  --img-root data/3mouse/images --json-file data/3mouse/annotations/dlc_shuffle0_val.json --out-img-root output_images --kpt-thr 0.0 && cp output_images/*.png data/


#python demo/bottom_up_img_demo.py configs/bottom_up/resnet/modelzoo/res50_rodent_512x512.py imagenet2rodents_train128/latest.pth  --img-root data/cocomagic/rodent_all/images/val --json-file data/cocomagic/rodent_all/annotations/val.json --out-img-root output_images --kpt-thr 0.0 && cp output_images/*.png data/

