import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.freqnet import freqnet
from options.test_options import TestOptions
import numpy as np
import random


DetectionTests = {
    # 'Original-Images': {
    #     'dataroot': '/home/ubuntu/Development/FreqNet-DeepfakeDetection/dataset/ForenSynths/cyclegan/apple',
    #     'no_resize': False,
    #     'no_crop': True,
    # },
    # 'Partially-Noised': {
    #     'dataroot': '/home/ubuntu/Development/FreqNet-DeepfakeDetection/dataset/perturbed-data/test/noise-50%',
    #     'no_resize': False,
    #     'no_crop': True,
    # },
    'Perturbed-Subset': {
        'dataroot': '/home/ubuntu/Development/FreqNet-DeepfakeDetection/dataset/perturbed-data/test/ForenSynths',
        'no_resize': False,
        'no_crop': True,
        'include_only': [
            'progan', 'stylegan', 'stylegan2', 'biggan',
            'cyclegan', 'stargan', 'gaugan', 'deepfake'
        ]
    },
}

opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# get model
model = freqnet(num_classes=1)

# from collections import OrderedDict
# from copy import deepcopy
# state_dict = torch.load(opt.model_path, map_location='cpu')['model']
# pretrained_dict = OrderedDict()
# for ki in state_dict.keys():
#     pretrained_dict[ki[7:]] = deepcopy(state_dict[ki])
# model.load_state_dict(pretrained_dict, strict=True)

model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.cuda()
model.eval()

for testSet in DetectionTests.keys():
    test_config = DetectionTests[testSet]
    dataroot = test_config['dataroot']
    include_only = test_config.get('include_only', None)
    printSet(testSet)

    accs = []
    aps = []
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    v_id = 0
    for val in sorted(os.listdir(dataroot)):
        if include_only and val not in include_only:
            continue  # skip folders not in the selected subset

        opt.dataroot = f'{dataroot}/{val}'
        opt.classes = ''  # os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = test_config['no_resize']
        opt.no_crop = test_config['no_crop']

        acc, ap, _, _, _, _ = validate(model, opt)
        accs.append(acc)
        aps.append(ap)
        print(f"({v_id} {val:12}) acc: {acc*100:.1f}; ap: {ap*100:.1f}")
        v_id += 1

    print(f"({v_id} {'Mean':10}) acc: {np.mean(accs)*100:.1f}; ap: {np.mean(aps)*100:.1f}")
    print('*' * 25)
