import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler


from tensorboardX import SummaryWriter
from sklearn import metrics
import numpy as np

import prismnet.model as arch
from prismnet import train, validate, log_print

from prismnet.model.utils import GradualWarmupScheduler
from prismnet.loader import SeqicSHAPE




def main():
    global writer, best_epoch
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch RBP Example')
    parser.add_argument('--datadir', type=str, default="data", help='data path')
    parser.add_argument('--testset', type=str, help='data path')
    parser.add_argument('--arch', default="PrismNet", help='data path')
    parser.add_argument('--lr_scheduler', default="cosine", help=' lr scheduler')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--log_interval', type=int, default=50, help='input batch size')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--beta', type=int, default=2, help='number of epochs to train for')
    
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='learning rate, default=0.001')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--seed', type=int, default=512, help='manual seed')
    parser.add_argument('--early_stopping', type=int, default=20, help='early stopping')
    
    parser.add_argument('--exp_name', type=str, default="cnn", metavar='N',
                        help='experiment name')
    parser.add_argument('--p_name', type=str, default="TIA1_Hela", metavar='N',
                        help='protein name')
    parser.add_argument('--eval', action='store_true', help='evaluation')
    parser.add_argument('--out_dir', type=str, default=".", help='output directory')
    parser.add_argument('--nstr', type=int, default=1, help='number of vector encoding for structure data')
    parser.add_argument('--type', type=str, default="pu", help='type')
    parser.add_argument('--tfboard', action='store_true', help='tf board')

    #parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
    #                    help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.001,
                        help='scale sparse rate (default: 0.0001)')
    args = parser.parse_args()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.type == 'pu':
        args.nstr = 1
        datatype = 'pu'
    else:
        args.nstr = 0
        datatype = 'seq'

    if args.tfboard:
        writer = SummaryWriter(args.out_dir)
    else:
        writer = None

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    data_path  = args.datadir + "/" + args.p_name + ".h5"
    model_path = args.out_dir +"/"+"ckpt_{}.pth"
    train_loader = torch.utils.data.DataLoader(SeqicSHAPE(data_path), \
        batch_size=args.batch_size, shuffle=True,  **kwargs)

    test_loader  = torch.utils.data.DataLoader(SeqicSHAPE(data_path, is_test=True), \
        batch_size=args.batch_size, shuffle=False, **kwargs)
    print("Train set:", len(train_loader.dataset))
    print("Test  set:", len(test_loader.dataset))


    print("Network Arch:", args.arch)
    model = getattr(arch, args.arch)(n_features=args.nstr+4)
    arch.param_num(model)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.beta))
    if args.eval:
        filename = model_path.format("best")
        # import pdb; pdb.set_trace()
        print("Loading model: {}".format(filename))
        model.load_state_dict(torch.load(filename))
        epoch = 0
        test_loss, acc, auc = validate(args, model, device, test_loader, epoch, criterion, writer)
        print("{} auc: {:.4f} acc: {:.4f}".format(args.p_name, auc, acc))
        return 

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=8, total_epoch=float(args.nepochs), after_scheduler=None)

    best_auc = 0
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, args.nepochs + 1):
        
        train(args, model, device, train_loader, optimizer,scheduler, epoch, criterion, writer)
        test_loss, acc, auc = validate(args, model, device, test_loader, epoch, criterion, writer)
        scheduler.step(epoch)
        color='green'
        if best_auc < auc:
            best_auc = auc
            best_acc = acc
            best_epoch = epoch
            color='red'
            filename = model_path.format("best")
            torch.save(model.state_dict(), filename)
        filename = model_path.format("latest")
        torch.save(model.state_dict(), filename)
        line='{} \t Test  Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} ({:.4f})'.format(\
            args.p_name, epoch, test_loss, acc, auc, best_auc)
        log_print(line, color=color, attrs=['bold'])
        if epoch - best_epoch > args.early_stopping:
            print("Early stop at %d, %s "%(epoch, args.exp_name))
            break
    print("{} auc: {:.4f} acc: {:.4f}".format(args.p_name, best_auc, best_acc))

if __name__ == '__main__':
    main()
