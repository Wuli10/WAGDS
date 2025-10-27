import torch.nn as nn
import torch
from torch.autograd import Variable
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from utils.viz import viz_pred_test1 as viz_pred_test
from torch.nn import functional as F

import argparse
from utils.evaluation import cal_kl, cal_sim, cal_nss
from utils.util import post_process_affordance_map

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data_path')
parser.add_argument('--phase', type=str, default='test')
parser.add_argument("--divide", type=str, default="Unseen") #"Seen" or "Unseen" 
parser.add_argument("--model_path", type=str, default="save_models_path") # the model weight path
parser.add_argument("--save_path", type=str, default="pred_results")
parser.add_argument("--crop_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument('--threshold', type=float, default='0.2')
# parser.add_argument("--init_weights", type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--viz', action='store_true', default=False)
parser.add_argument('--num_exo', type=int, default=3)
parser.add_argument('--n_layer', type=int, default=12)



args = parser.parse_args()

from models.EAGDS import MODEL

def normalize_map(atten_map):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val) / (max_val - min_val + 1e-10)

    return atten_norm



transform = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size), antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])

if args.divide == "Seen":
    # args.num_classes = 36
    aff_list = ['beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo",
                "talk_on", "text_on", "throw", "type_on", "wash", "write"]

    
elif args.divide=="Unseen":
    aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                     "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                     "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                     "swing", "take_photo", "throw", "type_on", "wash"]
else: # HICO-IIF
    aff_list = ['cut_with', 'drink_with', 'hold', 'open', 'pour', 'sip', 'stick', 'stir', 'swing', 'type_on']
    
args.test_root = os.path.join(args.data_root, args.divide, "testset", "egocentric")
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")

model, par = MODEL(args, num_classes=len(aff_list), pretrained=False, n=args.num_exo, n_layer=args.n_layer)
model.load_state_dict(torch.load(args.model_path))
model.eval()
model.cuda()


import datatest as datatest

testset = datatest.TrainData(egocentric_root=args.test_root, crop_size=args.crop_size, divide=args.divide, mask_root=args.mask_root)
MytestDataLoader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

masks = torch.load(args.divide+"_gt.t7")# GT_path
dict_1 = {}

KLs = []
SIMs = []
NSSs = []

KLs_cam = []
SIMs_cam = []
NSSs_cam = []

KLs_cam_refine = []
SIMs_cam_refine = []
NSSs_cam_refine = []

# KLs_cam_refine2 = []
# SIMs_cam_refine2 = []
# NSSs_cam_refine2 = []

# KLs_final_cam_refine2 = []
# SIMs_final_cam_refine2 = []
# NSSs_final_cam_refine2 = []

data_path = os.path.join(args.data_root, args.divide, "testset", "egocentric")


KLs_finalcam_dict = {}
KLs_cam_dict = {}
KLs_cam1_dict = {}
SIMs_cam_dict = {}
SIMs_final_cam_dict = {}
SIMs_cam1_dict = {}
NSSs_cam_dict = {}
NSSs_final_cam_dict = {}
NSSs_cam1_dict = {}



for step, (egocentric_image, label, mask_path, name) in enumerate(MytestDataLoader):
    label = label.cuda(non_blocking=True)
    egocentric_image = egocentric_image.cuda(
        non_blocking=True)
    cam_list, cam1_list = model.get(egocentric_image, label, name)
    for i in range(len(cam_list)):
        cam = cam_list[i]
        cam1 = cam1_list[i]


        cam = cam[0].cpu().detach().numpy()
        cam1 = cam1[0].cpu().detach().numpy()

        names = mask_path[0].split("/")
        key = names[-3] + "_" + names[-2] + "_" + names[-1]
        if key in masks.keys():
            gt = masks[key]
            mask = gt / 255.0
            
            # mask = cv2.resize(mask, (224, 224))

        image = Image.open(os.path.join(data_path, names[-3], names[-2], names[-1].replace(".png", ".jpg"))).convert('RGB')
        image = transform(image)
        
        # cam refined
        mask = cv2.resize(mask, (args.crop_size, args.crop_size))
        cam = normalize_map(cam)
        cam1 = normalize_map(cam1)
        cam1[cam <args.threshold]=0
        final_cam = normalize_map(cam+cam1)

        if i not in KLs_finalcam_dict:
            KLs_finalcam_dict[i] = []
            KLs_cam_dict[i] = []
            KLs_cam1_dict[i] = []
            SIMs_cam_dict[i] = []
            SIMs_cam1_dict[i] = []
            SIMs_final_cam_dict[i] = []
            NSSs_final_cam_dict[i] = []
            NSSs_cam_dict[i] = []
            NSSs_cam1_dict[i] = []
        
        kld = cal_kl(final_cam, mask)
        KLs_finalcam_dict[i].append(kld)
        kld = cal_kl(cam, mask)
        KLs_cam_dict[i].append(kld)
        kld = cal_kl(cam1, mask)
        KLs_cam1_dict[i].append(kld)

        sim = cal_sim(final_cam, mask)
        SIMs_final_cam_dict[i].append(sim)
        sim = cal_sim(cam, mask)
        SIMs_cam_dict[i].append(sim)
        sim = cal_sim(cam1, mask)
        SIMs_cam1_dict[i].append(sim)

        nss = cal_nss(final_cam, mask)
        NSSs_final_cam_dict[i].append(nss)
        nss = cal_nss(cam, mask)
        NSSs_cam_dict[i].append(nss)
        nss = cal_nss(cam1, mask)
        NSSs_cam1_dict[i].append(nss)




        aff, obj, name = names[-3], names[-2], names[-1].split('.')[0]


        if args.viz:
            
           
            viz_pred_test(args, image, final_cam, mask, aff, obj, name, type_=f'final_cam_{i}')
          

for i in range(len(KLs_finalcam_dict)):
    # cam_dict = KLs_cam_dict[i]
    # mKLD = sum(cam_dict) / len(cam_dict)
    # mSIM = sum(SIMs_cam_dict[i]) / len(SIMs_cam_dict[i])
    # mNSS = sum(NSSs_cam_dict[i]) / len(NSSs_cam_dict[i])
    # os.makedirs(args.save_path, exist_ok=True)
    # result_file = os.path.join(args.save_path, args.divide + f"_result_cam_{i}.txt")
    # with open(result_file, "w") as f:
    #     f.write(f"KLD = {round(mKLD, 3)}={sum(cam_dict)}/{len(cam_dict)}\nSIM = {round(mSIM, 3)}={sum(SIMs_cam_dict[i])}/{len(SIMs_cam_dict[i])}\nNSS = {round(mNSS, 3)}={sum(NSSs_cam_dict[i])}/{len(NSSs_cam_dict[i])}")
    #     print(f"KLD = {round(mKLD, 3)}={sum(cam_dict)}/{len(cam_dict)}\nSIM = {round(mSIM, 3)}={sum(SIMs_cam_dict[i])}/{len(SIMs_cam_dict[i])}\nNSS = {round(mNSS, 3)}={sum(NSSs_cam_dict[i])}/{len(NSSs_cam_dict[i])}")

    # cam1_dict = KLs_cam1_dict[i]
    # mKLD = sum(cam1_dict) / len(cam1_dict)
    # mSIM = sum(SIMs_cam1_dict[i]) / len(SIMs_cam1_dict[i])
    # mNSS = sum(NSSs_cam1_dict[i]) / len(NSSs_cam1_dict[i])
    # os.makedirs(args.save_path, exist_ok=True)
    # result_file = os.path.join(args.save_path, args.divide + f"_result_cam1_{i}.txt")
    # with open(result_file, "w") as f:
    #     f.write(f"KLD = {round(mKLD, 3)}={sum(cam1_dict)}/{len(cam1_dict)}\nSIM = {round(mSIM, 3)}={sum(SIMs_cam1_dict[i])}/{len(SIMs_cam1_dict[i])}\nNSS = {round(mNSS, 3)}={sum(NSSs_cam1_dict[i])}/{len(NSSs_cam1_dict[i])}")
    #     print(f"KLD = {round(mKLD, 3)}={sum(cam1_dict)}/{len(cam1_dict)}\nSIM = {round(mSIM, 3)}={sum(SIMs_cam1_dict[i])}/{len(SIMs_cam1_dict[i])}\nNSS = {round(mNSS, 3)}={sum(NSSs_cam1_dict[i])}/{len(NSSs_cam1_dict[i])}")

    final_cam_dict = KLs_finalcam_dict[i]
    mKLD = sum(final_cam_dict) / len(final_cam_dict)
    mSIM = sum(SIMs_final_cam_dict[i]) / len(SIMs_final_cam_dict[i])
    mNSS = sum(NSSs_final_cam_dict[i]) / len(NSSs_final_cam_dict[i])
    os.makedirs(args.save_path, exist_ok=True)
    result_file = os.path.join(args.save_path, args.divide + f"_result_final_cam_{i}.txt")
    with open(result_file, "w") as f:
        f.write(f"KLD = {round(mKLD, 3)}={sum(final_cam_dict)}/{len(final_cam_dict)}\nSIM = {round(mSIM, 3)}={sum(SIMs_final_cam_dict[i])}/{len(SIMs_final_cam_dict[i])}\nNSS = {round(mNSS, 3)}={sum(NSSs_final_cam_dict[i])}/{len(NSSs_final_cam_dict[i])}")
        print(f"KLD = {round(mKLD, 3)}={sum(final_cam_dict)}/{len(final_cam_dict)}\nSIM = {round(mSIM, 3)}={sum(SIMs_final_cam_dict[i])}/{len(SIMs_final_cam_dict[i])}\nNSS = {round(mNSS, 3)}={sum(NSSs_final_cam_dict[i])}/{len(NSSs_final_cam_dict[i])}")

    


    
    

    
    
    