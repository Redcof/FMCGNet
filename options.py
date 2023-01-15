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

        models = ["fmcgnet", ]

        ##
        # Base
        self.parser.add_argument('--model', help="path to output trained model", required=True)
        self.parser.add_argument('--plot_path', type=str, required=True, help="path to output loss/accuracy plot")
        self.parser.add_argument('--dataroot', default='', help='path to dataset')
        self.parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
        self.parser.add_argument('--path', default='', help='path to the folder or image to be predicted.')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--isize', type=int, default=128, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--device', type=str, default='cpu', help='Device: gpu | cpu | cuda')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='skipganomaly', choices=models,
                                 help='chooses which model to use.')
        self.parser.add_argument('--display_server', type=str, default="http://localhost",
                                 help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')
        self.parser.add_argument('--verbose', action='store_true', help='Print the training and model details.')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        self.parser.add_argument('--abnormal_class', default='', help='Anomaly class idx for mnist and cifar datasets')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')

        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console (per image).')
        self.parser.add_argument('--save_image_freq', type=int, default=100,
                                 help='frequency of saving real and fake images (per image).')
        self.parser.add_argument('--save_test_images', default='true', action='store_true',
                                 help='Save test images for demo (per image).')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='Weight for adversarial loss. default=1')
        self.parser.add_argument('--w_con', type=float, default=50, help='Weight for reconstruction loss. default=50')
        self.parser.add_argument('--w_lat', type=float, default=1, help='Weight for latent space loss. default=1')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        # ATZ dataset
        self.parser.add_argument('--atz_patch_db', default="", help='required. csv file path for atz patch dataset')
        self.parser.add_argument('--atz_train_txt', default=None, help='optional. text file path for atz '
                                                                       'train file dataset')
        self.parser.add_argument('--atz_test_txt', default=None,
                                 help='optional. text file path for atz train file dataset')
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

        # get gpu ids
        try:
            str_ids = self.opt.gpu_ids
            self.opt.gpu_ids = ast.literal_eval(str_ids)
        except Exception:
            self.opt.gpu_ids = []
        # if str(self.opt.gpu_ids).strip() != "":
        #     str_ids = self.opt.gpu_ids.split(',')
        #     self.opt.gpu_ids = []
        #     for str_id in str_ids:
        #         idx = int(str_id)
        #         if idx >= 0:
        #             self.opt.gpu_ids.append(idx)
        # else:
        #     self.opt.gpu_ids = []

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.verbose:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
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
