import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import sys
import os
from sklearn.metrics import roc_auc_score
import pandas as pd
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
sys.path.insert(0, './utils')
sys.path.insert(0, './models')

import data_utils as du
import test_utils as tu
import nodule_detector
from model_l5 import Net

"""
This script runs test over our validation set, which corresponds to the year 2.000 CT scan 
of each patient of the training dataset of ISBI Lung challenge 2018 (30 Patients).
"""

parser = argparse.ArgumentParser(description='PyTorch LungCancerPredictor Test')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--dicom_dir', default='./data/ISBI-deid-TRAIN/', help="path to dicom files")
parser.add_argument('--vols_dir', default='./data/patient_volumes/', help="path to patient volumes")
parser.add_argument('--cands_dir', default='./data/extracted_candidates/', help="path to extracted candidates")
parser.add_argument('--slices_dir', default='./data/slices/', help="path to nodule slices")
parser.add_argument('--sorted_slices_dir', default='./data/sorted_slices/', help="path to nodule sorted slices")
parser.add_argument('--sorted_slices_jpgs_dir', default='./data/sorted_slices_jpgs/', help="path to nodule sorted slices jpgs")

parser.add_argument('--resume', default='./models/model_predictor.pth', help="path to model of 5 lanes for test")
parser.add_argument('--csv', default='submission_test.csv', help="file name to save submission csv")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

ini_t_time = time.time()

print('creating patients...')
du.create_patients_from_dicom(args.dicom_dir, args.vols_dir)

print('extracting candidates...')
du.candidate_extraction(args.vols_dir, args.cands_dir)

print('creating candidate slices...')
du.create_candidate_slices(args.vols_dir, args.cands_dir, args.slices_dir)

print('running nodule detector...')
nodule_detector.run(args.slices_dir) #outputs scores_detector_test.npz

print('applying nms and sorting slices...')
Sc = np.load('scores_detector_test.npz')
tu.apply_nms(Sc, args.vols_dir, args.cands_dir, args.slices_dir, args.sorted_slices_dir, args.sorted_slices_jpgs_dir)

print('creating dataset...')
data, label = du.create_test_dataset(args.sorted_slices_dir, 1)
test = tu.prep_dataset(data, label)
test_loader = data_utils.DataLoader(test, batch_size=30, shuffle=False)
print('dataset ready!')

model = Net()
res_flag = 0
load_model = False

if args.resume != '': 
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.resume)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size() }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    res_flag = 1

if args.cuda:
    model.cuda()

if __name__ == '__main__':
    best_loss = None
    if load_model:
        best_loss = test(0)

output1 = np.load('preds3D_val.npy') #subtraction of year 2000 and 1999 masks + sum + sigmoid scores
epoch = 0
output, label = tu.test(epoch, model, test_loader, args) 
output = output.data.cpu().numpy()
PP = 0.68
outputx = output[:,1]*PP + output1[:,0]*(1-PP)

print('score(s): \n')
print(outputx)

if np.shape(data)[4] > 1:
    AUC = roc_auc_score(label, outputx)
    print('\n****************************')
    print('AUC: ' + str(AUC))
    print('****************************')

print('total elapsed time: %0.2f min\n' % ((time.time() - ini_t_time)/60.0))


