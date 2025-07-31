import os
import numpy as np
import cv2
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--divide', type=str, default='Seen')
parser.add_argument('--data_root', type=str, default='data_root')
args = parser.parse_args()

gt_root=f"{args.data_root}/{args.divide}/testset/GT/"
files=os.listdir(gt_root)
dict_1={}
for file in files:
    file_path=os.path.join(gt_root,file)
    objs=os.listdir(file_path)
    for obj in objs:
        obj_path=os.path.join(file_path,obj)
        images=os.listdir(obj_path)
        for img in images:
            img_path=os.path.join(obj_path,img)
            mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            key=file+"_"+obj+"_"+img
            dict_1[key]=mask
torch.save(dict_1, f"{args.divide}_gt.t7")