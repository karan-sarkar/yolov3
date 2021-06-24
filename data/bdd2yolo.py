import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from shutil import copyfile


def parse_arguments():
    parser = argparse.ArgumentParser(description='BDD100K to COCO format')
    parser.add_argument("--labels")
    parser.add_argument("--image_dir")
    parser.add_argument("--split")
    parser.add_argument(
          "-s", "--save_path",
          default="",
          help="path to save coco formatted label file",
    )
    parser.add_argument(
          "-a", "--attribute",
          default="timeofday",
          help="attribute to segment by",
    )
    parser.add_argument(
          "-f", "--flag",
          default="daytime",
          help="flag value for attribute to segment by",
    )
    return parser.parse_args()


args = parse_arguments()
if not os.path.exists(os.path.join(args.save_path, args.flag, args.split, 'images')):
    os.makedirs(os.path.join(args.save_path,  args.flag, args.split, 'images'))
if not os.path.exists(os.path.join(args.save_path,  args.flag, args.split, 'labels')):
    os.makedirs(os.path.join(args.save_path,  args.flag, args.split, 'labels'))

# create BDD training set detections in COCO format
print('Loading training set...')
with open(args.labels) as f:
    labeled_images = json.load(f)
print('Converting training set...')


attr_dict = dict()
attr_dict["categories"] = [
    {"supercategory": "none", "id": 1, "name": "person"},
    {"supercategory": "none", "id": 2, "name": "rider"},
    {"supercategory": "none", "id": 3, "name": "car"},
    {"supercategory": "none", "id": 4, "name": "bus"},
    {"supercategory": "none", "id": 5, "name": "truck"},
    {"supercategory": "none", "id": 6, "name": "bike"},
    {"supercategory": "none", "id": 7, "name": "motor"},
    {"supercategory": "none", "id": 8, "name": "traffic light"},
    {"supercategory": "none", "id": 9, "name": "traffic sign"},
    {"supercategory": "none", "id": 10, "name": "train"}
]

id_dict = {i['name']: i['id'] for i in attr_dict['categories']}



counter = 0
for i in tqdm(labeled_images):
    if i['attributes'][args.attribute] != args.flag:
        continue
    label_text = ''
    for label in i['labels']:
        if label['category'] in id_dict.keys():
            x1 = label['box2d']['x1']
            y1 = label['box2d']['y1']
            x2 = label['box2d']['x2']
            y2 = label['box2d']['y2']
            category_id = id_dict[label['category']]
            label_text += '\n' + str(category_id) + ' ' + str((x1 + x2) / 2 * 1280) + ' ' + str((y1 + y2) / 2 * 720) + ' ' + str((x2 - x1) / 1280) + ' ' + str((y2 - y1) / 720)
    copyfile(os.path.join(args.image_dir, i['name']), os.path.join(args.save_path,  args.flag, args.split, 'images', i['name']))
    if len(label_text) > 0:
        label_text = label_text[1:]
        name = i['name'][:-3] + 'txt'
        with open(os.path.join(args.save_path,  args.flag, args.split, 'labels', name), "w") as text_file:
            text_file.write(label_text)

    



    