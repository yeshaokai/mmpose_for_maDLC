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
    return_heatmap = False

    return_deconv_feature = True

    return_backbone_feature = True
    
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None


    deconv_feature_list = []
    backbone_feature_list = []
    
    # process each image
    for i in range(len(img_keys)):
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

        
        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')

        # show the results
        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=args.show,
            out_file=out_file)


        backbone_feature_list.append(returned_outputs['backbone_feature'])
        deconv_feature_list.append(returned_outputs['deconv_feature'])
    deconv_features = np.array(deconv_feature_list)
    backbone_features = np.array(backbone_feature_list)

    print ('deconv_features',deconv_features.shape)
    print ('backbone_features',backbone_features.shape)
    
    np.savez_compressed('inference_results',
                        deconv_features=deconv_features,
                        backbone_features = backbone_features
    )

if __name__ == '__main__':
    main()
