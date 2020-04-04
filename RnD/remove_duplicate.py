import os
from shutil import copyfile

out_path = "/home/vishal/datasets/val2017_/"
out_path_gt = "/home/vishal/datasets/val2017_gt/"
out_path_pred = "/home/vishal/datasets/val2017_pred/"

gt = os.listdir(out_path_gt)
pred = os.listdir(out_path)
print(len(gt), len(pred))

for name in gt:
    copyfile(os.path.join(out_path, name) , os.path.join(out_path_pred,name))