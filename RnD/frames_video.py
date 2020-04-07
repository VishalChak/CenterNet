import cv2
import os
import natsort
imgs_path = "/home/vishal/detection/CenterNet/data/out_2_hg/"
out_path = ""


flag = True
out = None
for file_name in natsort.natsorted(os.listdir(imgs_path)):
    img = cv2.imread(os.path.join(imgs_path, file_name))
    if flag:
        height, width, layers = img.shape
        size = (width,height)
        out = cv2.VideoWriter('/home/vishal/detection/CenterNet/data/center_net_hg_2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        out.write(img)
        flag = False
    else:
        out.write(img)
out.release()