"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """

'''
Example:
python3 tools/generate_hdf5.py \
        --gpu 0 \
        --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml \
        --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
        --out uitviic_val_resnet101_faster_rcnn_genome.tsv \
        --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel \
        --split /mnt/f9e5fe40-5d81-46dc-8450-9c1e67eff197/Projects/UIT-ViIC/val.json \
        --base-dir /mnt/f9e5fe40-5d81-46dc-8450-9c1e67eff197/Projects/UIT-ViIC
'''

import sys
import _init_paths # initialize path for faster_rcnn modules

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from tqdm import tqdm

import caffe
import argparse
import pprint
import os, sys
import base64
import numpy as np
import cv2
import csv
import random
import json

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 10
MAX_BOXES = 100

def load_image_ids(base_dir, split_name):
    split = []
    with open(split_name, "r") as file:
        data = json.load(file)
        for image in data["images"]:
            id = image["id"]
            filepath, filename = image["filepath"], image["filename"]
            img_dir = os.path.join(base_dir, filepath, filename)
            split.append((img_dir, id))

    return split
    
def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):

    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5'].data

    print(pool5.shape)

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
   
    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]),
        'features': base64.b64encode(pool5[keep_boxes])
    }   


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--base-dir", dest="base_dir", 
                        help="Base directory to the images' folder", type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

    
def generate_hdf5(net, image_ids, outfile):
  with open(outfile, 'a+') as tsvfile:
      writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
      for im_file, image_id in tqdm(image_ids):
        # writer.writerow(get_detections_from_im(net, im_file, image_id))
        get_detections_from_im(net, im_file, image_id)
     
if __name__ == '__main__':

    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    image_ids = load_image_ids(args.base_dir, args.data_split)
    random.seed(10)
    random.shuffle(image_ids)
    
    caffe.init_log()
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(args.prototxt, caffe.TEST, weights=args.caffemodel)

    generate_hdf5(net, image_ids, args.outfile)