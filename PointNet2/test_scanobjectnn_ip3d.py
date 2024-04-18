"""
Author: Tuo FENG
Date: 2024
"""
from data_utils.scanobjectnn import ScanObjectNNHardest
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from models.nearestcentroids_classier_scanobjectnn import NCC_LVQ21

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=15, type=int,  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default="pointnet2_cls_msg_scanobjectnn_ip3d", help='Experiment root')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--use_normals', default=False, help='use normals')
    return parser.parse_args()



def test(model, classier, loader, num_class=15, vote_num=1):
    test_pred = []
    test_true = []
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9, ncols=80):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            _, features, _ = classifier(points)
            feats, cosine_sim = classier(features.unsqueeze(-1), target.long())
            pred = torch.max(cosine_sim, dim=2)[0]
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = torch.argmax(pred, dim=1)

        test_pred.append(pred_choice.detach().cpu().numpy())
        test_true.append(target.cpu().numpy())

    test_pred = np.concatenate(test_pred)
    test_true = np.concatenate(test_true)
    oa= 100. * metrics.accuracy_score(test_true, test_pred)
    mAcc = 100. * metrics.balanced_accuracy_score(test_true, test_pred)
    print("OA: %.3f, mAcc: %.3f" % (oa, mAcc))
    return oa, mAcc



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = './data/ScanObjectNN/main_split/'

    test_dataset = ScanObjectNNHardest(data_dir=data_path, split='test')
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    classier = NCC_LVQ21(device='cuda:'+ str(args.gpu), num_classes = args.num_category, k=15, mu=0.999)

    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classier.load_state_dict(checkpoint['classier'])

    with torch.no_grad():
        oa, mAcc = test(classifier.eval(), classier, testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string("OA: %.3f, mAcc: %.3f" % (oa, mAcc))


if __name__ == '__main__':
    args = parse_args()
    main(args)