#!/usr/bin/env python
"""
# Author: XU Kui
# Created Time : 09 Nov 2020 11:14:31 PM CST
# Description:
    decription: x
"""
import os,sys
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import r2_score
# feature_list= ['AARS', 'AATF', 'ABCF1', 'AGGF1', 'AKAP1', 'AKAP8L', 'ALKBH5', 'APOBEC3C', 'AQR', 'ATXN2', 'AUH', 'BCCIP', 'BCLAF1', 'BUD13', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'CDC40', 'CPEB4', 'CPSF1', 'CPSF2', 'CPSF3', 'CPSF4', 'CPSF6', 'CPSF7', 'CSTF2', 'CSTF2T', 'DDX21', 'DDX24', 'DDX3X', 'DDX42', 'DDX51', 'DDX52', 'DDX55', 'DDX59', 'DDX6', 'DGCR8', 'DHX30', 'DKC1', 'DROSHA', 'EFTUD2', 'EIF3D', 'EIF3G', 'EIF3H', 'EIF4A3', 'eIF4AIII', 'EIF4G2', 'ELAVL1', 'EWSR1', 'EXOSC5', 'FAM120A', 'FASTKD2', 'FBL', 'FIP1L1', 'FKBP4', 'FMR1', 'FTO', 'FUS', 'FXR1', 'FXR2', 'G3BP1', 'GEMIN5', 'GNL3', 'GPKOW', 'GRWD1', 'GTF2F1', 'HLTF', 'HNRNPA1', 'HNRNPC', 'HNRNPD', 'HNRNPF', 'HNRNPK', 'HNRNPM', 'HNRNPU', 'HNRNPUL1', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'ILF3', 'KHDRBS1', 'KHSRP', 'LARP4', 'LARP7', 'LIN28A', 'LIN28B', 'LSM11', 'METAP2', 'METTL14', 'METTL3', 'MOV10', 'MTPAP', 'NCBP2', 'NIP7', 'NIPBL', 'NKRF', 'NOL12', 'NOLC1', 'NONO', 'NOP56', 'NOP58', 'NPM1', 'NUDT21', 'PABPC4', 'PABPN1', 'PCBP1', 'PCBP2', 'PHF6', 'PPIG', 'PRPF4', 'PRPF8', 'PTBP1', 'PTBP1PTBP2', 'PUM1', 'PUM2', 'PUS1', 'QKI', 'RBFOX2', 'RBM15', 'RBM22', 'RBM27', 'RBPMS', 'RPS11', 'RPS3', 'RTCB', 'SAFB2', 'SBDS', 'SDAD1', 'SERBP1', 'SF3A3', 'SF3B1', 'SF3B4', 'SLBP', 'SLTM', 'SMNDC1', 'SND1', 'SRRM4', 'SRSF1', 'SRSF7', 'SRSF9', 'SUB1', 'SUPV3L1', 'TAF15', 'TARDBP', 'TBRG4', 'TIA1', 'TIAL1', 'TNRC6A', 'TRA2A', 'TROVE2', 'U2AF1', 'U2AF2', 'U2AF65', 'UCHL5', 'UPF1', 'UTP18', 'UTP3', 'WDR3', 'WDR33', 'WDR43', 'WRN', 'WTAP', 'XRCC6', 'XRN2', 'YBX3', 'YTHDF2', 'YWHAG', 'ZC3H11A', 'ZC3H7B', 'ZNF622', 'ZNF800', 'ZRANB2']
feature_list= ['AARS', 'AATF', 'ABCF1', 'AGGF1', 'AKAP1', 'AKAP8L', 'ALKBH5', 'APOBEC3C', 'AQR', 'ATXN2', 'AUH', 'BCCIP', 'BCLAF1', 'BUD13', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'CDC40', 'CPEB4', 'CPSF1', 'CPSF2', 'CPSF3', 'CPSF4', 'CPSF6', 'CPSF7', 'CSTF2', 'CSTF2T', 'DDX21', 'DDX24', 'DDX3X', 'DDX42', 'DDX51', 'DDX52', 'DDX55', 'DDX59', 'DDX6', 'DGCR8', 'DHX30', 'DKC1', 'DROSHA', 'EFTUD2', 'EIF3D', 'EIF3G', 'EIF3H', 'EIF4A3',             'EIF4G2', 'ELAVL1', 'EWSR1', 'EXOSC5', 'FAM120A', 'FASTKD2', 'FBL', 'FIP1L1', 'FKBP4', 'FMR1', 'FTO', 'FUS', 'FXR1', 'FXR2', 'G3BP1', 'GEMIN5', 'GNL3', 'GPKOW', 'GRWD1', 'GTF2F1', 'HLTF', 'HNRNPA1', 'HNRNPC', 'HNRNPD', 'HNRNPF', 'HNRNPK', 'HNRNPM', 'HNRNPU', 'HNRNPUL1', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'ILF3', 'KHDRBS1', 'KHSRP', 'LARP4', 'LARP7', 'LIN28A', 'LIN28B', 'LSM11', 'METAP2', 'METTL14', 'METTL3', 'MOV10', 'MTPAP', 'NCBP2', 'NIP7', 'NIPBL', 'NKRF', 'NOL12', 'NOLC1', 'NONO', 'NOP56', 'NOP58', 'NPM1', 'NUDT21', 'PABPC4', 'PABPN1', 'PCBP1', 'PCBP2', 'PHF6', 'PPIG', 'PRPF4', 'PRPF8', 'PTBP1', 'PTBP1PTBP2', 'PUM1', 'PUM2', 'PUS1', 'QKI', 'RBFOX2', 'RBM15', 'RBM22', 'RBM27', 'RBPMS', 'RPS11', 'RPS3', 'RTCB', 'SAFB2', 'SBDS', 'SDAD1', 'SERBP1', 'SF3A3', 'SF3B1', 'SF3B4', 'SLBP', 'SLTM', 'SMNDC1', 'SND1', 'SRRM4', 'SRSF1', 'SRSF7', 'SRSF9', 'SUB1', 'SUPV3L1', 'TAF15', 'TARDBP', 'TBRG4', 'TIA1', 'TIAL1', 'TNRC6A', 'TRA2A', 'TROVE2', 'U2AF1', 'U2AF2', 'U2AF65', 'UCHL5', 'UPF1', 'UTP18', 'UTP3', 'WDR3', 'WDR33', 'WDR43', 'WRN', 'WTAP', 'XRCC6', 'XRN2', 'YBX3', 'YTHDF2', 'YWHAG', 'ZC3H11A', 'ZC3H7B', 'ZNF622', 'ZNF800', 'ZRANB2', 'eIF4AIII']
spec_list = ["AUH","HNRNPC","HNRNPU","IGF2BP1","IGF2BP3","LIN28B","SND1","TAF15","TIA1","FMR1","FXR1","FXR2","ILF3","KHDRBS1","KHSRP","PTBP1","TARDBP","TNRC6A","XRN2","BCLAF1","DDX6","EXOSC5","G3BP1","LARP4","NCBP2","PABPN1","PCBP1","SUPV3L1","UPF1","YBX3","PABPC4","PUM1","PUM2","SERBP1","HNRNPD","HNRNPF","QKI"]

top20_list = ['SND1', 'NPM1', 'KHDRBS1', 'GNL3', 'HNRNPUL1', 'TARDBP', 'ELAVL1', 'YTHDF2',
 'YBX3', 'LIN28B', 'YWHAG', 'ZC3H7B', 'TIA1', 'PUM2', 'RBFOX2', 'SERBP1', 'RBPMS',
 'RPS3', 'PUM1', 'PRPF8']
spec_top_list = ['AUH', 'BCLAF1', 'DDX6', 'ELAVL1', 'EXOSC5', 'FMR1', 'FXR1', 'FXR2', 'G3BP1', 'GNL3', 'HNRNPC', 'HNRNPD', 'HNRNPF', 'HNRNPU', 'HNRNPUL1', 'IGF2BP1', 'IGF2BP3', 'ILF3', 'KHDRBS1', 'KHSRP', 'LARP4', 'LIN28B', 'NCBP2', 'NPM1', 'PABPC4', 'PABPN1', 'PCBP1', 'PRPF8', 'PTBP1', 'PUM1', 'PUM2', 'QKI', 'RBFOX2', 'RBPMS', 'RPS3', 'SERBP1', 'SND1', 'SUPV3L1', 'TAF15', 'TARDBP', 'TIA1', 'TNRC6A', 'UPF1', 'XRN2', 'YBX3', 'YTHDF2', 'YWHAG', 'ZC3H7B']
import scipy.stats

import termplotlib as tpl
# from data_utils import load_data

import pickle
from sklearn import datasets, ensemble
# from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance



def plot(x, y, label="plot"):
    fig = tpl.figure()
    fig.plot(x, y, label=label, width=50, height=15)
    fig.show()

def plot_hist(sample,bins=40):
    counts, bin_edges = np.histogram(sample, bins=bins)
    fig = tpl.figure()
    fig.hist(counts, bin_edges, grid=[15, 25], orientation="horizontal",force_ascii=False)
    fig.show()

def normy(x):
    return (x-x.min())/(x.max() - x.min())

def normx(x):
    return 1/(1 + np.exp(-x.astype("float"))) 
    # return (x-x.mean())/x.std()

def get_topk_important_fea(filepath, topk=4):
    global feature_list
    feature_name=np.array(feature_list) 
    weight = np.load(filepath, allow_pickle=True)
    gain = weight['gain'].tolist()
    fea_gain = np.zeros(len(gain))
    for i in range(len(gain)):
        fea_gain[i] = gain['f'+str(i)]
    topk_flist = fea_gain.argsort()[::-1][:topk]
    
    
    return topk_flist


def get_topk_important_fea1(reg, topk=4):
    # global feature_list
    # feature_name=np.array(feature_list) 
    # weight = np.load(filepath, allow_pickle=True)
    # fscore = bst.get_fscore()
    feature_importance = reg.feature_importances_
    topk_flist = np.argsort(feature_importance)[::-1][:topk]
    return topk_flist

def get_topk_important_fea2(bst, topk=4):
    global feature_list
    feature_name=np.array(feature_list) 
    # weight = np.load(filepath, allow_pickle=True)
    # fscore = bst.get_fscore()
    fscore = bst.get_score(importance_type='gain')
    fea_fscore = np.zeros(len(fscore))
    for i in range(len(fscore)):fea_fscore[i] = fscore['f'+str(i)]
    topk_flist = fea_fscore.argsort()[::-1][:topk]
    return topk_flist



##
#  this script demonstrate how to fit generalized linear model in xgboost
#  basically, we are using linear model, instead of tree for our boosters

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=640, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=400, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--cv', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--train_data', default='', type=str,
                    help="path of the training data to use")
parser.add_argument('--test_data', default='', type=str,
                    help="path of the training data to use")
parser.add_argument('--pred_data', default='', type=str,
                    help="path of the training data to use")
parser.add_argument('--model_path', default='', type=str,
                    help="path of the training data to use")
parser.add_argument('--reg', default='squarederror', type=str,
                    help="path of the training data to use")
parser.add_argument('--booster', default='gbtree', type=str,
                    help="path of the training data to use")
parser.add_argument('--lam', type=int, default=-1, 
                    help='L2 reg (default: -1)'
                    )
parser.add_argument('--topk', type=int, default=0, 
                    help='topk features (default: 0)')
parser.add_argument('--randk', type=int, default=0, 
                    help='random k features (default: 0)')
parser.add_argument('--load_best', action='store_true', default=False,
                    help='load best model')
parser.add_argument('--fine_tune', action='store_true', default=False,
                    help='fine tuning ')
parser.add_argument('--cell_expr', action='store_true', default=False,
                    help='using cell expression')
parser.add_argument('--normx', action='store_true', default=False,
                    help='norm input data')
parser.add_argument('--plot', action='store_true', default=False,
                    help='norm input data')
parser.add_argument('--fsel', type=int, default=1, 
                    help='feature selector (default: -1)'
                    )
parser.add_argument('--sellist', type=int, default=0, 
                    help='feature selector (default: -1)'
                    )
args = parser.parse_args()

traindata = args.train_data
testdata = args.test_data
preddata = args.pred_data

if not os.path.exists(preddata):
    print(preddata," not found.")
    preddata = ""
if args.fine_tune:
    traindata = preddata.replace(".train.npz",".test.npz")
    print("Fine-tune on ",traindata)
    
print("Reading train data:",traindata)
print("Reading test data:",testdata)
print("Reading pred data:",preddata)

t_data = np.load(traindata,allow_pickle=True)
e_data = np.load(testdata,allow_pickle=True)
if preddata!="":
    p_data = np.load(preddata,allow_pickle=True)

t_x = t_data['x']
t_y = t_data['y']
e_x = e_data['x']
e_y = e_data['y']
if preddata!="":
    p_x = p_data['x']
    p_y = p_data['y']

print(" train X: min,max: {:.3f} {:.3f} {}".format(t_x.min(), t_x.max(), t_x.shape))
print(" train Y: min,max: {:.3f} {:.3f}".format(t_y.min(), t_y.max()))
print(" test  X: min,max: {:.3f} {:.3f} {}".format(e_x.min(), e_x.max(), e_x.shape))
print(" test  Y: min,max: {:.3f} {:.3f}".format(e_y.min(), e_y.max()))
if preddata!="":
    print(" pred  X: min,max: {:.3f} {:.3f}".format(p_x.min(), p_x.max(), p_x.shape))
    print(" pred  Y: min,max: {:.3f} {:.3f}".format(p_y.min(), p_y.max()))



norm_x = args.normx
if norm_x:
    t_x = normx(t_x)
    e_x = normx(e_x)
    if preddata!="":
        p_x = normx(p_x)

# plot_hist(t_y)
# print("-------------------------------------------")
norm_y = True
if norm_y:
    # t_y0 = np.zeros_like(t_y)
    # e_y0 = np.zeros_like(t_y)
    # import math
    # for i in range(t_y.shape[0]):
    #     t_y0[i] = math.log(t_y[i]+1)

    # for i in range(e_y.shape[0]):
    #     e_y0[i] = math.log(e_y[i]+1)
    # t_y = t_y0
    # e_y = e_y0
    # import pdb; pdb.set_trace()
    # t_y = np.log((t_y+1).astype("float"))
    # e_y = np.log((e_y+1).astype("float"))
    # t_x = abs(t_x)
    # e_x = abs(e_x)
    # t_y = np.log((t_y/2+2).astype("float"))
    # e_y = np.log((e_y/2+2).astype("float"))
    # t_x = t_x/10
    # e_x = e_x/10
    # t_y = np.log((t_y/2+2).astype("float"))
    # e_y = np.log((e_y/2+2).astype("float"))

    t_y = (t_y+1)/2
    e_y = (e_y+1)/2

    if preddata!="":
        # import pdb; pdb.set_trace()
        p_y = (p_y+1)/2


# plot_hist(t_y)
feature_name=np.array(feature_list) 
# if args.topk>0:
#     # filepath = args.model_path+"_weight_eval_test.npz"
#     # topk_list = get_topk_important_fea(filepath, topk=args.topk)
#     # feature_list = feature_name[topk_list]
#     # print("Using Top {} features: {}".format(args.topk, feature_name[topk_list]))
#     bst = xgb.Booster(model_file=args.model_path)
#     topk_list = get_topk_important_fea2(bst,args.topk)
#     feature_list = feature_name[topk_list]
#     t_x = t_x[:,topk_list]
#     e_x = e_x[:,topk_list]
#     if preddata!="":
#         p_x = p_x[:,topk_list]
#     print("Using Top {} features: {}".format(args.topk, feature_name[topk_list]))
#     # import pdb; pdb.set_trace()
#     args.model_path = args.model_path.replace("_best.model", "_topk{}_best.model".format(args.topk))
if args.sellist>0:
    if args.sellist==1:
        topk_list=[feature_list.index(p) for p in spec_list]

    elif args.sellist==2:
        topk_list=[feature_list.index(p) for p in spec_top_list]
    elif args.sellist==3: # top 20
        topk_list=[feature_list.index(p) for p in top20_list]
    else:
        raise "error no such list."


    feature_list = feature_name[topk_list]
    t_x = t_x[:,topk_list]
    e_x = e_x[:,topk_list]
    if preddata!="":
        p_x = p_x[:,topk_list]
    print("Using Top {} features: {}".format(args.topk, topk_list))
    print("Using Top {} features: {}".format(args.topk, feature_name[topk_list]))
    # import pdb; pdb.set_trace()
    # args.model_path = args.model_path.replace("_best.skl", "_topk{}_best.skl".format(args.topk))
    args.model_path = args.model_path.replace("_best.model", "_spec{}_best.model".format(args.sellist))
    

if args.topk>0:
    
    # filepath = args.model_path+"_weight_eval_test.npz"
    # topk_list = get_topk_important_fea(filepath, topk=args.topk)
    # feature_list = feature_name[topk_list]
    # print("Using Top {} features: {}".format(args.topk, feature_name[topk_list]))
    # bst = xgb.Booster(model_file=args.model_path)
    
    if args.fsel ==1:
        skl_model_path = args.model_path.replace("_best.model", "_best.skl")
        reg0 = pickle.load(open(skl_model_path, 'rb'))
        print("topk important_fea")
        topk_list = get_topk_important_fea1(reg0,args.topk)
    elif args.fsel ==2:
        skl_model_path = args.model_path.replace("_best.model", "_best.skl")
        reg0 = pickle.load(open(skl_model_path, 'rb'))
        print("topk permutation_importance")

        result = permutation_importance(reg0, e_x, e_y, n_repeats=10,
                                    random_state=42, n_jobs=2)
        topk_list = result.importances_mean.argsort()[::-1][:args.topk]#[::-1]
    else:
        print("topk gain")
        bst = xgb.Booster(model_file=args.model_path)
        topk_list = get_topk_important_fea2(bst,args.topk)


    feature_list = feature_name[topk_list]
    t_x = t_x[:,topk_list]
    e_x = e_x[:,topk_list]
    if preddata!="":
        p_x = p_x[:,topk_list]
    print("Using Top {} features: {}".format(args.topk, topk_list))
    print("Using Top {} features: {}".format(args.topk, feature_name[topk_list]))
    # import pdb; pdb.set_trace()
    # args.model_path = args.model_path.replace("_best.skl", "_topk{}_best.skl".format(args.topk))
    args.model_path = args.model_path.replace("_best.model", "_topk{}_best.model".format(args.topk))

if args.randk>0:
    # filepath = args.model_path+"_weight_eval_test.npz"
    topk_list = np.random.randint(171, size=args.randk)
    feature_list = feature_name[topk_list]
    t_x = t_x[:,topk_list]
    e_x = e_x[:,topk_list]
    if preddata!="":
        p_x = p_x[:,topk_list]
    print("Using Random {} features: {}".format(args.randk, feature_name[topk_list]))
    # print("Using Random {} features.".format(args.randk))
    args.model_path = args.model_path.replace("_best.model", "_randk{}_best.model".format(args.randk))

print(" train X: min,max: {:.3f} {:.3f} {}".format(t_x.min(), t_x.max(), t_x.shape))
print(" train Y: min,max: {:.3f} {:.3f}".format(t_y.min(), t_y.max()))
print(" test  X: min,max: {:.3f} {:.3f} {}".format(e_x.min(), e_x.max(), e_x.shape))
print(" test  Y: min,max: {:.3f} {:.3f}".format(e_y.min(), e_y.max()))
if preddata!="":
    print(" pred  X: min,max: {:.3f} {:.3f}".format(p_x.min(), p_x.max(), p_x.shape))
    print(" pred  Y: min,max: {:.3f} {:.3f}".format(p_y.min(), p_y.max()))


# dtrain = xgb.DMatrix(t_x, label=t_y, feature_names=feature_list)
# dtest  = xgb.DMatrix(e_x, label=e_y, feature_names=feature_list)
# if preddata!="":
#     dpred  = xgb.DMatrix(p_x, label=p_y, feature_names=feature_list)
# import pdb; pdb.set_trace()

dtrain = xgb.DMatrix(t_x, label=t_y)
dtest  = xgb.DMatrix(e_x, label=e_y)
if preddata!="":
    dpred  = xgb.DMatrix(p_x, label=p_y)
# change booster to gblinear, so that we are fitting a linear model
# alpha is the L1 regularizer
# lambda is the L2 regularizer
# you can also set lambda_bias which is L2 regularizer on the bias term
param = {'objective':'reg:squarederror', 'booster':'gbtree',"eval_metric": 'rmse',
          'lambda': 16,  'eta':0.1}
param = {'objective':'reg:'+args.reg, 'booster':args.booster,"eval_metric": 'rmse',
          'lambda': 16,  'eta':0.1}
print(param)
# normally, you do not need to set eta (step_size)
# XGBoost uses a parallel coordinate descent algorithm (shotgun),
# there could be affection on convergence with parallelization on certain cases
# setting eta to be smaller value, e.g 0.5 can make the optimization more stable
# param['eta'] = 1

##
# the rest of settings are the same
##
watchlist = [(dtrain, 'train'),(dtest, 'eval'), ]
num_round = 3000
best_r = 0
best_l = 0
best_p = 0


for la in range(0, 30, 2):
    if args.lam >= 0:
        param['lambda']=args.lam
    else:
        param['lambda'] = la
    print('lambda:', param['lambda'])
    
    early_stopping_rounds = 40
    if args.load_best:
        print("Loading best model.")
        bst = xgb.Booster(model_file=args.model_path)
        # topk_list = get_topk_important_fea2(bst,args.topk)
        # feature_list = feature_name[topk_list]
        # print("Using Top {} features22: {}".format(args.topk, feature_name[topk_list]))
        # import pdb; pdb.set_trace()
        # bst.save_model(args.model_path)
        if args.fine_tune:
            print("Fine tuning.")
            early_stop = xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                metric_name='rmse', 
                save_best=True,
                data_name='eval'
            )
            bst = xgb.train(param, dtrain, num_round, watchlist, callbacks=[early_stop],)
            args.model_path = args.model_path.replace("_best.model", "_finetune_best.model")
    elif args.cv:
        nfold = 5
        print("Do Cross Validation: {} fold.".format(nfold))
        param['verbosity']=1
        hist = xgb.cv(param, dtrain, num_round, 
            nfold=nfold, 
            verbose_eval=True,
            early_stopping_rounds=early_stopping_rounds)
        print(hist)
    else:
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name='rmse', 
            save_best=True,
            data_name='eval'
        )
        bst = xgb.train(param, dtrain, num_round, watchlist, callbacks=[early_stop],)
    bst.save_model(args.model_path)
    gain = bst.get_score(importance_type='gain')
    total_gain = bst.get_score(importance_type='total_gain')

    e_preds = bst.predict(dtest)
    e_labels = dtest.get_label()
    r, p = scipy.stats.pearsonr(e_labels, e_preds)    
    # r2=r2_score(labels, preds)
    print("Test R: {:f}, R^2: {:f}, P-value: {:e}".format(r, r**2, p))
    # if preddata!="":
    #     r,p = predict(bst, dtest)
    if r> best_r:
        best_r = r
        best_p = p
        best_l = la
        print("### -> Best...")

    if preddata!="":
        # print("dpred: ",p_x.shape)
        p_preds = bst.predict(dpred)
        p_labels = dpred.get_label()
        r, p = scipy.stats.pearsonr(p_labels, p_preds)    
        print("Pred R: {:f}, R^2: {:f}, P-value: {:e}".format(r, r**2, p))
    else:
        p_labels=None
        p_preds=None


    np.savez_compressed(args.model_path+"_weight_eval_test.npz", 
        gain=gain, 
        total_gain=total_gain,
        eval_label=e_labels,
        eval_pred=e_preds,
        test_label=p_labels,
        test_pred=p_preds,
        )
    if args.lam >= 0: # pred
        sys.exit(0)

print("Best la: {}\nR: {:f}, R^2: {:f}, P-value: {:e}".format(best_l, best_r, best_r**2, p))
# print("Best R: {:f}, R^2: {:f}, lambda: {}".format(best_r, best_r**2, best_l))
    # gain = bst.get_score(importance_type='gain')
    # total_gain = bst.get_score(importance_type='total_gain')
    # np.savez_compressed("low_fi_{:d}.npz".format(la), gain=gain, total_gain=total_gain)
    # xgb.plot_importance(bst,importance_type='gain', max_num_features=20)
    # plt.savefig("fi_{:d}.pdf".format(la))
    # print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
