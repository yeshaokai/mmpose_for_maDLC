export CUDA_VISIBLE_DEVICES=1
python demo/bottom_up_video_demo.py configs/bottom_up/resnet/modelzoo//res50_rodent_512x512.py data/modelzoo_checkpoints/rodent.pth \
       --video-path data/rodent.mp4\
       --out-video-root output_video
