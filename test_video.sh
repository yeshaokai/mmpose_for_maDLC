#python demo/bottom_up_video_demo.py  configs/bottom_up/resnet/coco/res50_coco_512x512.py checkpoints/res50_coco_512x512-5521bead_20200816.pth\
#       --video-path data/3mouse/videocompressed0.mp4\
#       --out-video-root videocompressed0_test.mp4
python demo/bottom_up_video_demo.py configs/bottom_up/resnet/3mouse/res50_3mouse_512x512.py work_dirs/res50_3mouse_512x512/epoch_300.pth \
       --video-path data/videocompressed4.mp4\
       --out-video-root output_video
