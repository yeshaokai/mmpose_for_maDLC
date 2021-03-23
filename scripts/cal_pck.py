import numpy as np
import json
from scipy.optimize import linear_sum_assignment

result = 'work_dirs/res50_mpii_512x512/result_keypoints.json'
#result = '/mmpose/work_dirs/res50_rodents_512x512_imagenet/result_keypoints.json'

annotation = 'data/mpii/annotations/val.json'
#annotation = '/mmpose/data/modelzoo/rodent/annotations/val.json'

from munkres import Munkres,print_matrix

def build_container_result(lst):
    container = {}
    for obj in lst:
        image_id = obj['image_id']

        keypoints = obj['keypoints']
        if image_id not in container:
            container[image_id] = {}
            container[image_id]['keypoints'] = []
        container[image_id]['keypoints'].append(keypoints)
    return container


def build_container_gt(lst,ids):
    container = {}
    annotations = lst['annotations']
        

    
    for anno in annotations:
        image_id = anno['image_id']
        if image_id not in ids:
            continue
        keypoints = anno['keypoints']
        x,y,width,height = anno['bbox']
        diagnal = np.sqrt(width**2+height**2)
        if image_id not in container:
            container[image_id] = {}
            container[image_id]['keypoints'] = []
            container[image_id]['diagnal'] = []
        container[image_id]['keypoints'].append(keypoints)
        container[image_id]['diagnal'].append(diagnal)
        
    return container

with open (result,'r') as f:
    res = json.load(f)


    
res = build_container_result(res)

ids = set(res.keys())


with open(annotation,'r') as f:
    gt = json.load(f)

    
gt = build_container_gt(gt,ids)


def load_result_and_gt():
    pass

def _convert(individuals):
    # from xyv 2 nw h

    n = len(individuals)                          

    num_joints = 16
    
    nwh = np.zeros((n,num_joints,2))
    
    for person_id,person in enumerate(individuals):

        kpt_num = len(person)//3
        for idx in range(kpt_num):
            offset = idx*3
            x = person[offset]
            y = person[offset+1]
            v = person[offset+2]

            nwh[person_id][idx][0] = x
            nwh[person_id][idx][1] = y


            
    return nwh

def calc_dists(preds, target, scales):

    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if scales[n] == 0:
                dists[c,n] = - 1
                continue
            
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:

                dists[c, n] = np.linalg.norm(preds[n,c,:] - target[n,c,:])/scales[n]
            else:

                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)

    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def get_pairwise_m(res,gt):
    # k,n1,n2
    k = gt.shape[1] # k= num_joints
    n1 = res.shape[0]
    n2 = gt.shape[0]
    m = np.zeros((k,n1,n2))

    for i in range(m.shape[0]): # joint_id
        for j in range(m.shape[1]): # person in res
            for q in range(m.shape[2]): # persons in gt
                res_coord = res[j][i]
                gt_coord = gt[q][i]
                d = np.linalg.norm(res_coord-gt_coord)
                m[i][j][q] = d
    m = m.mean(axis=0)

    return m
    
    
def convert_xyv2nwh(res,gt):

    gt_ids = list(gt.keys())
    acc_sum = 0
    for idx in gt_ids:
        res_persons = res[idx]['keypoints']
        gt_persons = gt[idx]['keypoints']
        diagnals = gt[idx]['diagnal']

                
        N = len(gt_persons) # max_num
        
        res_nwh = _convert(res_persons)
        gt_nwh = _convert(gt_persons)

        #print (res_nwh.shape)
        #print (gt_nwh.shape)
        
        #refer_kpts = [0,1]
        refer_kpts = [2,13] # mpii

        m = get_pairwise_m(res_nwh,gt_nwh)
        
        tmp = linear_sum_assignment(m)

        tmp = np.array(tmp).astype(int).T
        
        permu_res  = np.zeros(gt_nwh.shape)

        for row,col in tmp:
            permu_res[col] = res_nwh[row]
                        
        res_nwh = permu_res

        scales = np.array([np.linalg.norm(gt_nwh[i,refer_kpts[0],:]-gt_nwh[i,refer_kpts[1],:]) for i in range(gt_nwh.shape[0])])

        #scales = np.array(diagnals)
        dists = calc_dists(res_nwh,gt_nwh,scales)

        acc = dist_acc(dists)
        acc_sum+=acc
    acc_sum/=len(gt_ids)
    print (acc_sum)


convert_xyv2nwh(res,gt)
        


