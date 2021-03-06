###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model

from option import Options

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable


class Trainer():
    def __init__(self, args):
        self.args = args
        # data transforms
        if args.dataset == 'spacenet3':
            input_transform = transform.Compose([
                transform.ToTensor(),
                transform.Normalize([30.24584637, 32.54369452, 36.74206311],
                                    [5.94954947, 5.04209975, 4.56778059])])
        elif args.dataset == 'spacenet8':
            input_transform = transform.Compose([
                transform.ToTensor()])
        else:
            input_transform = transform.Compose([
                transform.ToTensor(),
                transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size, 'vAOI': args.vAOI}
        trainset = get_dataset(args.dataset, split=args.train_split, mode='train',
                               **data_kwargs)
        testset = get_dataset(args.dataset, split='val', mode='val',
                              **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class
        # model
        if args.dataset == 'spacenet8':
            model = get_segmentation_model(args.model, dataset=args.dataset,
                                           backbone=args.backbone, aux=args.aux,
                                           se_loss=args.se_loss, norm_layer=SyncBatchNorm,
                                           base_size=args.base_size, crop_size=args.crop_size,
                                           input_channels=8, multi_res_loss=args.multi_res_loss)
        else:
            model = get_segmentation_model(args.model, dataset=args.dataset,
                                           backbone=args.backbone, aux=args.aux,
                                           se_loss=args.se_loss, norm_layer=SyncBatchNorm,
                                           base_size=args.base_size, crop_size=args.crop_size,
                                           input_channels=3, multi_res_loss=args.multi_res_loss)
        # print(model)
        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr}, ]
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
        optimizer = torch.optim.SGD(params_list, lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        # criterions
        self.criterion = SegmentationLosses(se_loss=args.se_loss,
                                            aux=args.aux,
                                            nclass=self.nclass,
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight,
                                            multi_res_loss=args.multi_res_loss,
                                            multi_res_weight=args.multi_res_weight)
        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()
        # resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader))
        self.best_pred = 0.0
        self.best_mF1 = 0.0
        self.best_pixAcc = 0.0
        self.best_mIoU = 0.0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader)
        for i, (image, target) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            outputs = self.model(image)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)

    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target.cuda()
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            mF1 = utils.batch_f1_score(pred.data, target, self.nclass)
            return correct, labeled, inter, union, mF1

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        total_f1 = 0
        tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union, mF1 = eval_batch(self.model, image, target)
            else:
                with torch.no_grad():
                    correct, labeled, inter, union, mF1 = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f, mF1: %.3f' % (pixAcc, mIoU, mF1))
            total_f1 += mF1

        total_f1 /= len(self.valloader)
        new_pred = (pixAcc + mIoU + total_f1)/2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if self.best_pixAcc < pixAcc:
            self.best_pixAcc = pixAcc
        if self.best_mIoU < mIoU:
            self.best_mIoU = mIoU
        if self.best_mF1 < total_f1:
            self.best_mF1 = total_f1
        print('\nBest: pixAcc: {0:1.4f} | mIoU: {1:1.4f} | mF1: {2:1.4f} |\n'.format(self.best_pixAcc,
                                                                                     self.best_mIoU,
                                                                                     self.best_mF1))
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, self.args, is_best)


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    if args.eval:
        trainer.validation(trainer.args.start_epoch)
    else:
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                trainer.validation(epoch)
