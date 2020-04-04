import sys
CENTERNET_PATH = "/home/vishal/detection/CenterNet/src/lib/"
sys.path.insert(0, CENTERNET_PATH)

import torch
import _init_paths
from detectors.detector_factory import detector_factory 
from opts import opts
from utils.debugger import Debugger
import glob
torch.backends.cudnn.deterministic = True

MODEL_PATH = "/home/vishal/detection/CenterNet/models/ctdet_coco_dla_2x.pth"
TASK = 'ctdet'
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
opt.debug = 1
detector = detector_factory[opt.task](opt)
img = "a.png"
#img = "000007.jpg"
# ret = detector.run(img)
# results = ret['results']
# #print(results)
import os
#data_path = "/home/vishal/detection/CenterNet/data/VOC/"
#data_path_voc = "/home/vishal/datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/"
data_path_coco = "/home/vishal/datasets/val2017/"

for i in os.listdir(data_path_coco):
    # print(os.path.join(data_path_coco, i))
    ret = detector.run(os.path.join(data_path_coco, i))

