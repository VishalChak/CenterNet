import json


file_path = "/home/vishal/datasets/annotations_trainval2017/annotations/instances_val2017.json"
# file_path = "/home/vishal/datasets/annotations_trainval2014/annotations/instances_val2014.json"
#file_path = "/home/vishal/datasets/instances_train-val2014/annotations/instances_val2014.json"

f = open(file_path,)

data = json.load(f)
print(data['annotations'][0].keys())
print(data['annotations'][100])

out_path = "/home/vishal/datasets/val2017_gt/"
skip = [12, 26, 29, 30, 45, 66, 68,69, 71, 83, 91]
# coco_class_name = [
#      'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#      'bus', 'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
#      'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
#      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
#      'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
#      'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass',
#      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#      'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake',
#      'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
#      'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
#      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#      'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
# ]

coco_names_91 = [

"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic_light",
"fire_hydrant","street_sign","stop_sign","parking_meter","bench","bird","cat","dog","horse","sheep",
"cow","elephant","bear","zebra","giraffe","hat","backpack","umbrella","shoe","eye_glasses","handbag",
"tie","suitcase","frisbee","skis","snowboard","sports_ball","kite","baseball_bat","baseball_glove",
"skateboard","surfboard","tennis_racket","bottle","plate","wine_glass","cup","fork","knife","spoon",
"bowl","banana","apple","sandwich","orange","broccoli","carrot","hot_dog","pizza","donut","cake","chair",
"couch","potted_plant","bed","mirror","dining_table","window","desk","toilet","door","tv","laptop","mouse",
"remote","keyboard","cell_phone","microwave","oven","toaster","sink","refrigerator","blender","book","clock",
"vase","scissors","teddy_bear","hair_drier","toothbrush","hair_brush"


]

#print(len(coco_names_91))
a = []
import os
for node in data['annotations']:
    if int(node['category_id']) not in skip:
        file_name = str(node['image_id']).zfill(12) + ".txt"
        f_gt = open(os.path.join(out_path, file_name), 'a+')
        arr = []
        arr.append(coco_names_91[node['category_id']-1])
        node['bbox'][2] = node['bbox'][2] + node['bbox'][0]
        node['bbox'][3] = node['bbox'][3] + node['bbox'][1] 
        arr.extend(int(i) for i in node['bbox'])
        arr.append('\n')
        print(file_name)
        f_gt.write(" ".join(str(i) for i in arr))
        # print(coco_class_name[node['category_id']] , node['bbox'], node['id'], node['category_id'])
        # print(str(node['image_id']).zfill(12))
        f_gt.close()

#print(max(a) - len(skip), len(coco_class_name))
#print(sorted(a))
