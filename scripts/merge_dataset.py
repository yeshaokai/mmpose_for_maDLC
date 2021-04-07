#category merge
#skeleton merge (if same)
#handle multiple animal, if exists
#show number of images in each dataset and decide the best sampling


animals = ['atrwtiger','stanfordextra','dog_cat_sheep_horse_cow','badja']
sample_ratios = [1.0,1.0,1.0,1.0]
import json
import numpy as np
import os

def get_json_objs(animal):
    root = os.path.join(animal,'annotations')
    train = os.path.join(root,'train.json')
    val = os.path.join(root,'val.json')
    with open(train,'r') as f:
        train_obj = json.load(f)
    with open(val,'r') as f:
        val_obj = json.load(f)
    return train_obj,val_obj

def generate_animal_category():
    ret = {}    
    ret['keypoints'] = ['nose','left_eye', 'right_eye', 'left_ear', 'right_ear', 'throat', 'withers', 'tailset', 'left_front_paw', 'right_front_paw', 'left_front_wrist', 'right_front_wrist', 'left_front elbow', 'right_front_elbow', 'left_back_paw', 'right_back_paw', 'left_back_hock', 'right_back_hock', 'left_back_stifle', 'right_back_stifle']
    ret['supercategory'] = 'animal'
    ret['name'] = 'animal'
    ret['skeleton'] = [[15, 17], [17, 19], [16, 18], [18, 20], [19, 8], [20, 8], [8, 7], [7, 6], [6, 1], [1, 2], [1, 3], [2, 3], [2, 4], [3, 5], [6, 13], [6, 14], [13, 11], [11, 9], [14, 12], [12, 10]]
    ret['id'] = 1
    return ret

def abs_path(root,img_name,stage):
    return os.path.join(root,'images',stage,img_name)

def convert2_abs_paths(imgs,animal,stage):
    for img in imgs:
        img['file_name'] = abs_path(animal,img['file_name'],stage)


def sample_N(obj,N):
    imgid_2_anno = {}
    imgid_2_img = {}
    annotations = obj['annotations']
    images = obj['images']        
    categories = obj['categories']

    for img in images:
        id = img['id']

        imgid_2_img[id] = img


    if N is None:
        N = len(imgid_2_img)
        
    for anno in annotations:
        image_id = anno['image_id']                
        if image_id not in imgid_2_anno:
            imgid_2_anno[image_id] = []

        imgid_2_anno[image_id].append(anno)
            

    ret_categories = categories
    ret_imgs = []
    ret_annos = []
    
    for id in list(imgid_2_anno.keys())[:N]:
        ret_imgs.append(imgid_2_img[id])
        for anno in imgid_2_anno[id]:
            ret_annos.append(anno)
                
    ret_obj = {}
    ret_obj['images'] = ret_imgs

    ret_obj['annotations'] = ret_annos
    ret_obj['categories'] = ret_categories
    return ret_obj


def update_imageids(objs):
    # check the total id from the previous one
    # shift all image_id by the offsetl
    # shift all anno id by the same offset
    offset = 0
    for i,obj in enumerate(objs):
        id_to_new_id = {}
        ori_id_to_new_id = {}

        # new ids
        for j,img_obj in enumerate(obj['images']):
            imgid = img_obj['id']
            new_id = offset+j
            ori_id_to_new_id[imgid] = new_id
            img_obj['id'] = new_id

        # correct anno id
        for anno_obj in obj['annotations']:
            imgid = anno_obj['image_id']
            new_id = ori_id_to_new_id[imgid]
            anno_obj['image_id'] = new_id
                    
        offset += len(obj['images'])
        
        
def merge(objs,animals,stage):
    # merge all
    update_imageids(objs)
    
    ret = {}
    images = []
    annotations = []
    categories = [] # this one should be shared

    for obj,animal in zip(objs,animals):
        convert2_abs_paths(obj['images'],animal,stage)
        images.extend(obj['images'])
        annotations.extend(obj['annotations'])
    ret['images'] = images
    ret['annotations'] = annotations
    ret['categories'] = [generate_animal_category()]

    return ret

train_obj_list = []
val_obj_list = []

for animal,ratio in zip(animals,sample_ratios):
    train_obj,val_obj = get_json_objs(animal)

    n = int(len(train_obj['images'])*ratio)
    
    sampled_train_obj = sample_N(train_obj,n)

    print ('sampled {} from animal {}'.format(n,animal))
    
    sampled_val_obj = sample_N(val_obj,100)

    print ('{} in {}_train'.format(len(sampled_train_obj['images']),animal))    
    print ('{} in {}_val'.format(len(sampled_val_obj['images']),animal))

    
    train_obj_list.append(sampled_train_obj)
    val_obj_list.append(sampled_val_obj)
    

    
train_merged_obj = merge(train_obj_list,animals,'train')
val_merged_obj = merge(val_obj_list,animals,'val')

print ('merged_images',len(train_merged_obj['images']))
print ('merged_annotations',len(train_merged_obj['annotations']))

'''
print ('train')
print (train_merged_obj)
print ('val')
print (val_merged_obj)
'''

with open('all_animal_train.json','w') as f:
    json.dump(train_merged_obj,f)


with open('all_animal_val.json','w') as f:
    json.dump(val_merged_obj,f)    
