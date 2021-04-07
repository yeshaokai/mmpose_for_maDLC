export CUDA_VISIBLE_DEVICES=0,1,2

bash  tools/dist_train.sh  configs/bottom_up/resnet/modelzoo/res50_superanimal_512x512.py 
