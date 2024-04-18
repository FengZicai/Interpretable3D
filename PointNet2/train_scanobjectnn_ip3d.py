"""
Author: Tuo FENG
Date: 2024
"""

import os 
import random
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse
import sklearn.metrics as metrics

from pathlib import Path
from tqdm import tqdm
from data_utils.scanobjectnn_npc import ScanObjectNNHardest
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from models.nearestcentroids_classier_scanobjectnn import NCC_LVQ21


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_msg_ip3d', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=15, type=int,  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_msg_scanobjectnn_ip3d', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
        
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    pytorch_device='cuda:'+ str(args.gpu)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = './data/ScanObjectNN/main_split/'

    train_dataset = ScanObjectNNHardest(data_dir=data_path, split='train')
    test_dataset = ScanObjectNNHardest(data_dir=data_path, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=False)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    classier = NCC_LVQ21(device=pytorch_device, num_classes = args.num_category, k=15, mu=0.999)

    if not args.use_cpu:
        classifier = classifier.to(pytorch_device)
        criterion = criterion.to(pytorch_device)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = 0
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classier.load_state_dict(checkpoint['classier'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif  args.optimizer == 'Adamw':
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_oa = 0.0
    best_macc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        test_pred = []
        test_true = []
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9, ncols=80):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shuffle_points(points[:, :, 0:3])

            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.to(pytorch_device), target.to(pytorch_device)

            pred, features, trans_feat = classifier(points)
            feats, cosine_sim = classier(features.unsqueeze(-1), target)
            logits = torch.max(cosine_sim, dim=2)[0]

            loss = criterion(logits, target.long(), trans_feat)
            loss += criterion(pred, target.long(), trans_feat)
            # pred_choice = predict(cosine_sim)
            logits = torch.max(cosine_sim, dim=2)[0]
            pred_choice = torch.argmax(logits, dim=1)
            
            test_pred.append(pred_choice.detach().cpu().numpy())
            test_true.append(target.cpu().numpy())

            loss.backward()
            optimizer.step()

            classier.batch_training_update(feats.detach(), cosine_sim.detach(),target.long())

            global_step += 1
        scheduler.step()

        test_pred = np.concatenate(test_pred)
        test_true = np.concatenate(test_true)
        oa = 100. * metrics.accuracy_score(test_true, test_pred)
        log_string('Train OA: %f' % oa)

        with torch.no_grad():
            oa, mAcc =  test_macc_oa(classifier.eval(), classier, testDataLoader, num_class=num_class)
            if (oa >= best_oa):
                best_oa = oa
                best_epoch = epoch + 1

            if (mAcc >= best_macc):
                best_macc = mAcc
            log_string('Test OA: %f, mAcc: %f' % (oa, mAcc))
            log_string('Best OA: %f, mAcc: %f' % (best_oa, best_macc))

            if (oa >= best_oa):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'oa': oa,
                    'mAcc': mAcc,
                    'model_state_dict': classifier.state_dict(),
                    'classier': classier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test_macc_oa(model, classier, loader, num_class=15):
    test_pred = []
    test_true = []

    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9, ncols=80):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        _, features, _ = classifier(points)
        _, cosine_sim = classier(features.unsqueeze(-1), target.long())
        logits = torch.max(cosine_sim, dim=2)[0]
        pred_choice = torch.argmax(logits, dim=1)

        test_pred.append(pred_choice.detach().cpu().numpy())
        test_true.append(target.cpu().numpy())

    test_pred = np.concatenate(test_pred)
    test_true = np.concatenate(test_true)
    oa= 100. * metrics.accuracy_score(test_true, test_pred)
    mAcc = 100. * metrics.balanced_accuracy_score(test_true, test_pred)
    print("OA: %.3f, mAcc: %.3f" % (oa, mAcc))
    return oa, mAcc
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
