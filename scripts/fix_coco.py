# fix the category id thing

import json
import numpy as np
import os
from shutil import copyfile

root = 'dog_cat_sheep_horse_cow/annotations'


def file_transfer(train_imgs,val_imgs):
    root = 'dog_cat_sheep_horse_cow/images/'
    import glob as glob
    #imgs = glob.glob(src+'/*.j*g')
    for img in train_imgs:
        img_name = img['file_name']
        src = os.path.join(root,'backup',img_name)
        dest = os.path.join(root,'train',img_name)
        copyfile(src, dest)

    for img in val_imgs:
        img_name = img['file_name']
        src = os.path.join(root,'backup',img_name)
        dest = os.path.join(root,'val',img_name)
        copyfile(src, dest)        

with open(root+'/'+'animal_keypoints.json','r') as f:

    obj = json.load(f)
    imgid_2_anno = {}
    imgid_2_img = {}    
    annotations = obj['annotations']
    for anno in annotations:
        anno['category_id'] = 1
    images = obj['images']
    categories = obj['categories']
    
    for img in images:
        id = img['id']

        imgid_2_img[id] = img
    
    for anno in annotations:
        image_id = anno['image_id']                
        if image_id not in imgid_2_anno:
            imgid_2_anno[image_id] = []

        imgid_2_anno[image_id].append(anno)
            
    ret_categories = categories

    N = len(imgid_2_img.keys())

    train_N = int(N*0.8)

    train_imgs = []
    train_annos = []

    for id in list(imgid_2_anno.keys())[:train_N]:
        train_imgs.append(imgid_2_img[id])
        for anno in imgid_2_anno[id]:
            train_annos.append(anno)
                
    train_obj = {}
    train_obj['images'] = train_imgs
    train_obj['annotations'] = train_annos
    train_obj['categories'] = ret_categories


    val_imgs = []
    val_annos = []


    for id in list(imgid_2_anno.keys())[train_N:]:
        val_imgs.append(imgid_2_img[id])
        for anno in imgid_2_anno[id]:
            val_annos.append(anno)

    val_obj = {}            
    val_obj['images'] = val_imgs
    val_obj['annotations'] = val_annos
    val_obj['categories'] = ret_categories    


file_transfer(train_imgs,val_imgs)


'''    
with open(os.path.join(root,'train_fix.json'),'w') as f:
    json.dump(train_obj,f)


with open(os.path.join(root,'val_fix.json'),'w') as f:
    json.dump(val_obj,f)    
'''
