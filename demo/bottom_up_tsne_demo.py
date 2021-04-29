import os
from argparse import ArgumentParser

from xtcocotools.coco import COCO

from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)

import numpy as np

def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    parser.add_argument('--tag',type=str,help='for tag use')
    
    args = parser.parse_args()

    #assert args.show or (args.out_img_root != '')

    coco = COCO(args.json_file)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    print (dataset)
    #assert (dataset == 'BottomUpCocoDataset')

    img_keys = list(coco.imgs.keys())

    # optional
    return_heatmap = True

    return_deconv_feature = True
    return_backbone_feature = True
    
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None


    deconv_feature_list = []
    backbone_feature_list = []
    heatmap_list = []
    
    sample_N = 5
    
    # process each image
    for i in range(sample_N):

        if i == len(img_keys)-1:
            break
        image_id = img_keys[i]

        
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])

        print (image_name)
        # test a single image, with a list of bboxes.        
        
        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            image_name,
            return_heatmap=return_heatmap,
            return_backbone_feature = return_backbone_feature,
            return_deconv_feature = return_deconv_feature,
            outputs=output_layer_names)    

        # show the results


        for e in returned_outputs:

            heatmap = e['heatmap']
            print (heatmap.shape)
            
            deconv_feature = e['deconv_feature'][0].detach().cpu()
            
            backbone_feature = e['backbone_feature'][0].detach().cpu()
            
            deconv_num_feature = deconv_feature.shape[1]
            backbone_num_feature = backbone_feature.shape[1]
            heatmap_num_feature = heatmap.shape[1]
            
            backbone_feature_reordered = np.transpose(backbone_feature,(0,2,3,1)).reshape(-1,backbone_num_feature)

            deconv_feature_reordered = np.transpose(deconv_feature,(0,2,3,1)).reshape(-1,deconv_num_feature)

            heatmap_reordered = np.transpose(heatmap,(0,2,3,1)).reshape(-1,heatmap_num_feature)
            
            backbone_feature_list.append(backbone_feature_reordered)
            deconv_feature_list.append(deconv_feature_reordered)
            heatmap_list.append(heatmap_reordered)

    deconv_features = np.concatenate(deconv_feature_list,axis=0)
    backbone_features = np.concatenate(backbone_feature_list,axis=0)
    heatmaps = np.concatenate(heatmap_list,axis=0)

    
    print ('deconv_features',deconv_features.shape)
    print ('backbone_features',backbone_features.shape)
    print ('heatmaps',heatmaps.shape)
    
    np.savez_compressed('{}_features'.format(args.tag),
                        deconv_features=deconv_features,
                        backbone_features = backbone_features,
                        heatmaps=heatmaps
    )

if __name__ == '__main__':
    main()
