'''
Credits to Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation

Adopted from https://github.com/yulequan/UA-MT

With Modification

'''

import os
import argparse
import torch
from networks.vnet import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default=None, help='saved .pth filename')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../evaluation/"
test_save_path = "../evaluation/img/"+os.path.basename(FLAGS.model)+"/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]


def test_calculate_metric():
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = FLAGS.model
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(FLAGS.model)
    print(metric)