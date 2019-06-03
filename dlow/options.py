import argparse
import os
import pickle as pkl

import torch


def create_sub_dirs(opt, sub_dirs):
    for sub_dir in sub_dirs:
        dir_path = os.path.join(opt.expr_dir, sub_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        setattr(opt, sub_dir, dir_path)


class TrainOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment
        #         self.parser.add_argument('--name', type=str, required=True, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--name', type=str, default='first',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/',
                                 help='models are saved here')

        # data
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')

        # exp
        self.parser.add_argument('--seed', type=int, default=0, help='manual seed')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # training
        self.parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--niter', type=int, default=25, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=25,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # model
        self.parser.add_argument('--ngf', type=int, default=16, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=16, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--nlatent', type=int, default=16,
                                 help='# of latent code dimensions. Used only for stochastic models, e.g. cycle_ali')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet', help='selects model to use for netG')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--max_gnorm', type=float, default=500.,
                                 help='max grad norm to which it will be clipped (if exceeded)')

        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')

        # monitoring
        self.parser.add_argument('--monitor_gnorm', type=bool, default=True, help='flag set to monitor grad norms')
        self.parser.add_argument('--display_freq', type=int, default=500,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--eval_A_freq', type=int, default=1, help='frequency of evaluating on A')
        self.parser.add_argument('--eval_B_freq', type=int, default=1, help='frequency of evaluating on B')

        self.initialized = True

    def parse(self, sub_dirs=None):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # Set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.opt.expr_dir = expr_dir

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        file_name = os.path.join(expr_dir, 'opt.pkl')
        with open(file_name, 'wb') as opt_file:
            pkl.dump(args, opt_file)

        # create sub dirs
        if sub_dirs is not None:
            create_sub_dirs(self.opt, sub_dirs)

        return self.opt
