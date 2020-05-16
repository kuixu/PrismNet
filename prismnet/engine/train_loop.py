from __future__ import print_function
import argparse, os
import numpy as np
import torch
import torch.nn as nn

import prismnet.model as arch
from prismnet.utils import log_print
from prismnet.model.utils import compute_acc_auc

    
def train(args, model, device, train_loader, optimizer, scheduler, epoch, criterion, writer):
    # scheduler.step(epoch)
    # lr = scheduler.get_lr()[0]
    lr = optimizer.param_groups[0]['lr']
    model.train()
    NUM_BATCH = len(train_loader.dataset)//args.batch_size
    acc_lst = []
    auc_lst = []
    total_loss = 0
    for batch_idx, (x0, y0) in enumerate(train_loader):
        x, y = x0.float().to(device), y0.to(device).float()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        prob = torch.sigmoid(output)
        acc_, auc_ = compute_acc_auc(prob, y)
        total_loss += loss.item()
        acc_lst.append(acc_)
        auc_lst.append(auc_)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     line = '{} \t Train Epoch: {} {:2.0f}% Loss: {:.4f} '.format(\
        #     args.p_name, epoch, 100.0 * batch_idx / len(train_loader), loss.item())
        #     print(line)
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
    total_loss /= NUM_BATCH
    acc = 100. * np.mean(acc_lst)
    auc = np.mean(auc_lst)

    line='{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} lr: {:.6f}'.format(\
         args.p_name, epoch, total_loss, acc, auc, lr)#scheduler.get_lr()[0])
    log_print(line, color='green', attrs=['bold'])

    if args.tfboard:
        writer.add_scalar('loss/train', total_loss, epoch)
        writer.add_scalar('acc/train', acc, epoch)
        writer.add_scalar('AUC/train', auc, epoch)
        writer.add_scalar('lr', lr, epoch)

def validate(args, model, device, test_loader, epoch, criterion, writer):
    model.eval()
    test_loss = 0
    NUM_BATCH = len(test_loader.dataset)//args.batch_size
    acc_lst = []
    auc_lst = []
    with torch.no_grad():
        for batch_idx, (x0, y0) in enumerate(test_loader):
            x, y = x0.float().to(device), y0.to(device).float()
            output  = model(x)
            loss = criterion(output, y)
            prob = torch.sigmoid(output)
            acc_, auc_ = compute_acc_auc(prob, y)
            test_loss += loss.item() # sum up batch loss
            acc_lst.append(acc_)
            auc_lst.append(auc_)
    test_loss /= NUM_BATCH
    acc = 100. * np.mean(acc_lst)
    auc = np.mean(auc_lst)

    
    if args.tfboard:
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('acc/test', acc, epoch)
        writer.add_scalar('AUC/test', auc, epoch)
    return test_loss, acc, auc

def inference(args, model, device, test_loader):
    model.eval()
    acc_lst = []
    auc_lst = []
    with torch.no_grad():
        for batch_idx, (x0, y0) in enumerate(test_loader):
            x, y = x0.float().to(device), y0.to(device).float()
            output  = model(x)
            acc_, auc_ = compute_acc_auc(output, y)
            acc_lst.append(acc_)
            auc_lst.append(auc_)
    acc = 100. * np.mean(acc_lst)
    auc = np.mean(auc_lst)

    return acc, auc

