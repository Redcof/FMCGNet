""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import ast
import configparser
import os
import pathlib

import torch


# pylint: disable=C0103,C0301,R0903,W0622

class Options:
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataroot',
                                 default='/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/'
                                         'THZ_dataset_det_VOC/JPEGImages',
                                 help='path to dataset')
        self.parser.add_argument('--batchsize', type=int, default=256, help='input batch size')
        self.parser.add_argument('--dilation', type=int, default=1, help='Dilation value for LeNet models')
        self.parser.add_argument('--deformable', action='store_true', default=False,
                                 help='Enable deformable convolution layers or not')
        self.parser.add_argument('--ff', action='store_true', default=False,
                                 help='Enable forward-forward algorithm or not')
        self.parser.add_argument('--loss', type=str, default='CE', choices=['CE', 'SFL', 'BCE'],
                                 help='CEL:Categorical Cross-Entropy Loss, SFL: Sigmoid focal loss,'
                                      ' BCEL: Balanced Cross-entropy loss')
        self.parser.add_argument('--isize', type=int, default=128, choices=[128, 64], help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'],
                                 help='Device: gpu | cpu ')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--dataset', type=str, default='atz', help='name of the dataset')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=47, type=int, help='manual seed')
        self.parser.add_argument('--print_freq', default=5, type=int, help='Log printing frequency')
        ##
        # Train
        self.parser.add_argument('--phase', type=str, default='train', choices=["train", "val", "test"],
                                 help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=2, help='number of epochs to train for')
        # ATZ dataset
        self.parser.add_argument('--atz_patch_db',
                                 default="../customdataset/atz/atz_patch_dataset__3_128_36_v2_10%_30_99%.csv",
                                 help='required. csv file path for atz patch dataset')
        self.parser.add_argument('--atz_wavelet',
                                 default="{'wavelet':'sym4', 'method':'VisuShrink','level':3, 'mode':'hard'}",
                                 help='required. csv file path for atz patch dataset')
        self.parser.add_argument('--atz_patch_overlap', default=0.2, help='Patch overlap')
        self.parser.add_argument('--atz_wavelet_denoise', action="store_true",
                                 help='Flag to perform wavelet based denoise')
        self.parser.add_argument('--atz_classes', default=[], help='Specify a list of classes for experiment.'
                                                                   'Example: ["KK", "CK", "CL"]')
        self.parser.add_argument('--atz_subjects', default=[], help='Specify a list of subjects for experiment.'
                                                                    'Example: ["F1", "M1", "F2"]')
        self.parser.add_argument('--area_threshold', default=0.1, type=float,
                                 help='Object area threshold in percent to consider the patch as anomaly or not.'
                                      'Values between 0 and 1. where 0 is 0%, 1 is 100%, and 0.5 is 50% and so on.'
                                      'Default: 10% or 0.1')
        self.parser.add_argument('--atz_ablation', default=0, type=int,
                                 help='Used for ablation experiment. Patches with [no-threat:threat=n:n].')

        self.isTrain = True
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count()
        try:
            cuda_current = torch.cuda.current_device()
        except:
            cuda_current = "cpu"
        print("CUDA Info", cuda_available, cuda_count, cuda_current)

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        # saving
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.name, self.opt.dataset)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        from datetime import datetime

        # datetime object containing current date and time
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        setattr(self.opt, 'log', self.log)
        return self.opt

    def log(self, str_):
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('%s\n' % str_)
        print(str_)

    @staticmethod
    def mission_control(section, key):
        inifile = str(pathlib.Path(__file__).parents[0] / 'mission_control.ini')
        config = configparser.ConfigParser()
        config.read(inifile)
        return config[section][key]
