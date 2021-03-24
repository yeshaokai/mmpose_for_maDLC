# load the all data json file
import json
import numpy as np

with open('mouse_keypoints.json','r') as f:
    all_data = json.load(f)

for shuffle in [0,1,2]:
    
    dlc_split = '../3mouse_shuffule{}.json'.format(shuffle)

    with open(dlc_split,'r') as f:
        dlc_split = json.load(f)

    train_imgs = dlc_split['train_data']
    test_imgs = dlc_split['test_data']

    # split the data to train/split according to  DLC

    imgid_2_anno = {}
    imgid_2_img = {}
    img_2_imgid = {}

    annotations = all_data['annotations']

    for anno in annotations:
        anno['category_id'] = 1
    
    images = all_data['images']
    categories = all_data['categories']

    for img in images:
        id = img['id']
        filename = img['file_name']
        imgid_2_img[id] = img
        img_2_imgid[filename] = id

    for anno in annotations:
        image_id = anno['image_id']
        if image_id not in imgid_2_anno:
            imgid_2_anno[image_id] = []
        imgid_2_anno[image_id].append(anno)

    ret_train_imgs = []
    ret_train_annotations = []
    ret_train_categories = categories
    for imgname in train_imgs:
        imgid = img_2_imgid[imgname]
        ret_train_imgs.append(imgid_2_img[imgid])
        ret_train_annotations.extend(imgid_2_anno[imgid])

    ret_train_obj = {}
    ret_train_obj['images'] = ret_train_imgs
    ret_train_obj['annotations'] = ret_train_annotations
    ret_train_obj['categories'] = ret_train_categories
        
    ret_val_imgs = []
    ret_val_annotations = []
    ret_val_categories = categories
    for imgname in test_imgs:
        imgid = img_2_imgid[imgname]
        ret_val_imgs.append(imgid_2_img[imgid])
        ret_val_annotations.extend(imgid_2_anno[imgid])    

    ret_val_obj = {}
    ret_val_obj['images'] = ret_val_imgs
    ret_val_obj['annotations'] = ret_val_annotations

    ret_val_obj['categories'] = ret_val_categories

        
    assert len(ret_train_imgs) == len(train_imgs)
    assert len(ret_val_imgs) == len(test_imgs)


    
    print ('len(ret_train_annotations)')
    print (len(ret_train_annotations))

    print ('len(ret_val_imgs)')
    print (len(ret_val_imgs))
    
    print ('len(ret_val_annotations)')
    print (len(ret_val_annotations))


    with open('dlc_shuffle{}_train.json'.format(shuffle), 'w') as f:
        json.dump(ret_train_obj,f)
    with open('dlc_shuffle{}_val.json'.format(shuffle), 'w') as f:
        json.dump(ret_val_obj,f)        



