import json

with open('data/modelzoo/rodent/annotations/train.json','r') as f:
    gt = json.load(f)

def parse(f):
    imgid_2_anno = {}
    imgid_2_img = {}
    annotations = f['annotations']
    images = f['images']
    
    print (len(images))
    print (len(annotations))
    
    categories = f['categories']

    for img in images:
        id = img['id']

        imgid_2_img[id] = img
    
    for anno in annotations:
        image_id = anno['image_id']                
        if image_id not in imgid_2_anno:
            imgid_2_anno[image_id] = []

        imgid_2_anno[image_id].append(anno)
            


    ret_categories = categories



    split = [4,16,32,64,128]

    for num in split:
        ret_imgs = []
        ret_annos = []
        
        for id in list(imgid_2_anno.keys())[:num]:
            ret_imgs.append(imgid_2_img[id])
            for anno in imgid_2_anno[id]:
                ret_annos.append(anno)
                
        ret_obj = {}
        ret_obj['images'] = ret_imgs
        ret_obj['annotations'] = ret_annos
        ret_obj['categories'] = ret_categories

        with open('data/modelzoo/rodent/annotations/train_{}.json'.format(num),'w') as f:
            json.dump(ret_obj,f)

    
    
    
    
parse(gt)
