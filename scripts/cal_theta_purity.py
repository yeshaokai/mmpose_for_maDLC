import json
import numpy as np
from sklearn import metrics
from scipy.spatial import cKDTree

def purity_score(y_true,y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true,y_pred)
    return np.sum(np.amax(contingency_matrix,axis=0))/np.sum(contingency_matrix)

def _find_closest_neighbors(xy_true, xy_pred, k=5):
    n_preds = xy_pred.shape[0]
    tree = cKDTree(xy_pred)
    dist, inds = tree.query(xy_true, k=k)
    idx = np.argsort(dist[:, 0])
    neighbors = np.full(len(xy_true), -1, dtype=int)
    picked = set()
    for i, ind in enumerate(inds[idx]):
        for j in ind:
            if j not in picked:
                picked.add(j)
                neighbors[idx[i]] = j
                break
        if len(picked) == n_preds:
            break
    return neighbors


#result_path = 'work_dirs/res50_3mouse_512x512/result_keypoints.json'
result_path = 'work_dirs/hrnet_w32_marmoset_512x512/result_keypoints.json'
gt_path = 'data/marmoset/annotations/marmoset_keypoints.json'


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


purities = []
for img_id in mydict:
    dts = mydict[img_id]['res']

    dt_x = []
    dt_y = []
    gt_x = []
    gt_y = []
    gts = mydict[img_id]['gt']
    dists = np.zeros((len(dts),len(gts)))
    
    preds = []
    for i,dt in enumerate(dts):
        for _ in range(len(dt)//3):
            preds.append(i)

    gt_ids = []
    for (id,gt) in enumerate(gts):
        
        g = np.array(gt)
        xg = g[0::3]; yg = g[1::3];
        for _ in xg:
            gt_ids.append(id)
        for u in xg:
            gt_x.append(u)            
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

    xy_true = [(gt_x[i],gt_y[i]) for i in range(len(gt_x))]
    xy_pred = [(dt_x[i],dt_y[i]) for i in range(len(dt_x))]

    xy_true = np.array(xy_true)
    xy_pred = np.array(xy_pred)
    
    neighbors = _find_closest_neighbors(xy_true,xy_pred)
    valid = neighbors !=-1
    

    id_gt = xy_true[valid]
    id_hyp = xy_pred[neighbors[valid]]

    purity = purity_score(id_gt,id_hyp)
    print (imgid_2_filename[img_id],purity)
    purities.append(purity)


#with open('purity_results.csv','w') as f:
purities = np.array(purities)
np.savetxt('marmoset_purity_results.csv',purities)
