import json_tricks as json
import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from mmpose.datasets.builder import DATASETS
from .bottom_up_coco import BottomUpCocoDataset


@DATASETS.register_module()
class BottomUpMarmosetDataset(BottomUpCocoDataset):
    """MHPv2.0 dataset for top-down pose estimation.

    `The Multi-Human Parsing project of Learning and Vision (LV) Group,
    National University of Singapore (NUS) is proposed to push the frontiers
    of fine-grained visual understanding of humans in crowd scene.
    <https://lv-mhp.github.io/>`


    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    MHP keypoint indexes::

        0: "right ankle",
        1: "right knee",
        2: "right hip",
        3: "left hip",
        4: "left knee",
        5: "left ankle",
        6: "pelvis",
        7: "thorax",
        8: "upper neck",
        9: "head top",
        10: "right wrist",
        11: "right elbow",
        12: "right shoulder",
        13: "left shoulder",
        14: "left elbow",
        15: "left wrist",

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        super(BottomUpCocoDataset,self).__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        #self.ann_info['flip_index'] = [
        #    5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10
        #]

        self.ann_info['flip_index'] = range(15)
        

        self.ann_info['use_different_joint_weights'] = False
        #self.ann_info['joint_weights'] = np.array(
        #    [
                #1.5, 1.2, 1., 1., 1.2, 1.5, 1., 1., 1., 1., 1.5, 1.2, 1., 1.,
                #1.2, 1.5
        #    ],
        #    dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        self.ann_info['joint_weights'] = np.ones(15)
        
        # Adapted from COCO dataset
        self.sigmas = np.ones(15)*0.1
        

        self.coco = COCO(ann_file)

        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            (self._class_to_coco_ind[cls], self._class_to_ind[cls])
            for cls in self.classes[1:])
        self.img_ids = self.coco.getImgIds()
        if not test_mode:
            self.img_ids = [
                img_id for img_id in self.img_ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'marmoset'

        print(f'=> num_images: {self.num_images}')

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        with open(res_file, 'r') as file:
            res_json = json.load(file)
            if not res_json:
                info_str = list(zip(stats_names, [
                    0,
                ] * len(stats_names)))
                return info_str

        coco_det = self.coco.loadRes(res_file)

        coco_eval = COCOeval(
            self.coco, coco_det, 'keypoints', self.sigmas, use_area=False)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

