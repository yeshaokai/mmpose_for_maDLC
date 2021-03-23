import json
import numpy as np
from sklearn import metrics
def purity_score(y_true,y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true,y_pred)
    return np.sum(np.amax(contingency_matrix,axis=0))/np.sum(contingency_matrix)


result_path = 'work_dirs/res50_3mouse_512x512/result_keypoints.json'
gt_path = 'data/3mouse/annotations/val.json'


with open(result_path) as f:
    res = json.load(f)
with open(gt_path) as f:
    gt = json.load(f)


mydict = {}

imgid_2_filename = {}

for img in gt['images']:
    image_id = img['id']
    filename = img['file_name']
    imgid_2_filename[image_id] = filename
        

for e in res:

    image_id, keypoints = e['image_id'],e['keypoints']
    if image_id not in mydict:
        mydict[image_id] = {}
    if 'res' not in mydict[image_id]:
        mydict[image_id]['res'] = []
    mydict[image_id]['res'].append(keypoints)


    
for ann in gt['annotations']:

    image_id = ann['image_id']

    if 'gt' not in mydict[image_id]:
        mydict[image_id]['gt'] = []

        
    mydict[image_id]['gt'].append(ann['keypoints'])




for img_id in mydict:
    dts = mydict[img_id]['res']

    dt_x = []
    dt_y = []
    gt_x = []
    gt_y = []
    gts = mydict[img_id]['gt']
    dists = np.zeros((len(dts),len(gts)))
    for (j,gt) in enumerate(gts):

        g = np.array(gt)
        xg = g[0::3]; yg = g[1::3];

        #print(xg,yg)
        for i,dt in enumerate(dts):
            d = np.array(dt)
            xd = d[0::3]; yd = d[1::3]
            
            dx = xd-xg
            dy = yd-yg

            dist = np.sqrt(np.sum(dx**2+dy**2))
            dists[i,j] = dist
            
    preds_ids = np.argmin(dists,axis=0)
    preds = []
    for i,dt in enumerate(dts):
        for _ in range(len(dt)//3):
            preds.append(preds_ids[i])

    gt_ids = []
    for (id,gt) in enumerate(gts):
        
        g = np.array(gt)
        xg = g[0::3]; yg = g[1::3];
        for u in xg:
            gt_x.append(u)
            gt_ids.append(id)
        for u in yg:
            gt_y.append(u)

    gt_ids = np.array(gt_ids)
    for (_,dt) in enumerate(dts):
        d = np.array(dt)
        xd = d[0::3]; yd = d[1::3]
        for u in xd:
            dt_x.append(u)
        for u in yd:
            dt_y.append(u)

    point_dists = np.zeros((len(dt_x),len(gt_x)))
    for i in range(len(dt_x)):
        for j in range(len(gt_x)):
            dx = dt_x[i] - gt_x[j]
            dy = dt_y[i] - gt_y[j]

            point_dists[i,j] = np.sqrt(dx**2+dy**2)
    #print (point_dists)
    # pointwise comparison

    gts = gt_ids[np.argmin(point_dists,axis=0)]
    preds = np.array(preds)

    print ('gts')
    print (gts)
    print ('preds')
    print (preds)
    
    
    purity = purity_score(gts,preds)
    #print (imgid_2_filename[img_id],purity)
    
    
