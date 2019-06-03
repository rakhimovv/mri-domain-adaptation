import functools
import os
from collections import OrderedDict

import networks
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def criterion_GAN(pred, target_is_real, use_sigmoid=True):
    if use_sigmoid:
        if target_is_real:
            target_var = Variable(pred.data.new(pred.size()).long().fill_(1.))  # real = 1
        else:
            target_var = Variable(pred.data.new(pred.size()).long().fill_(0.))  # fake = 0

        loss = F.binary_cross_entropy(pred, target_var)
    else:
        if target_is_real:
            target_var = Variable(pred.data.new(pred.size()).fill_(1.))
        else:
            target_var = Variable(pred.data.new(pred.size()).fill_(0.))

        loss = F.mse_loss(pred, target_var)

    return loss


def discriminate(net, crit, fake, real):
    pred_fake = net(fake)
    loss_fake = crit(pred_fake, False)

    pred_true = net(real)
    loss_true = crit(pred_true, True)

    return loss_fake, loss_true, pred_fake, pred_true


class DLOW(object):
    """Domain Flow for Adaptation and Generalization"""

    def __init__(self, opt, testing=False):

        ##### model options
        self.old_lr = opt.lr
        opt.use_sigmoid = opt.no_lsgan

        self.opt = opt

        ##### define all networks we need here
        self.netZ_2_LAT = networks.define_Z_2_LAT(1, opt.nlatent, opt.gpu_ids)

        self.netG_A_B = networks.define_stochastic_G(nlatent=opt.nlatent, input_nc=opt.input_nc,
                                                     output_nc=opt.output_nc, ngf=opt.ngf,
                                                     # ngf = of gen filters in first conv layer
                                                     which_model_netG=opt.which_model_netG,
                                                     norm=opt.norm, use_dropout=opt.use_dropout,
                                                     # norm = instance or batch (default instance)
                                                     gpu_ids=opt.gpu_ids)

        self.netG_B_A = networks.define_stochastic_G(nlatent=opt.nlatent, input_nc=opt.input_nc,
                                                     output_nc=opt.output_nc, ngf=opt.ngf,
                                                     # ngf = # of gen filters in first conv layer
                                                     which_model_netG=opt.which_model_netG,
                                                     norm=opt.norm, use_dropout=opt.use_dropout,
                                                     # norm = instance or batch (default instance)
                                                     gpu_ids=opt.gpu_ids)

        self.netD_A = networks.define_D_A(input_nc=opt.input_nc,
                                          ndf=opt.ndf, norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)

        self.netD_B = networks.define_D_B(input_nc=opt.output_nc,
                                          ndf=opt.ndf, norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)

        ##### define all optimizers here
        self.optimizer_Z_2_LAT = torch.optim.Adam(self.netZ_2_LAT.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G_A = torch.optim.Adam(self.netG_B_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G_B = torch.optim.Adam(self.netG_A_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr / 5., betas=(opt.beta1, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr / 5., betas=(opt.beta1, 0.999))

        self.criterionGAN = functools.partial(criterion_GAN, use_sigmoid=opt.use_sigmoid)
        self.criterionCycle = F.l1_loss

        if not testing:
            with open("%s/nets.txt" % opt.expr_dir, 'w') as nets_f:
                networks.print_network(self.netG_A_B, nets_f)
                networks.print_network(self.netG_B_A, nets_f)
                networks.print_network(self.netD_A, nets_f)
                networks.print_network(self.netD_B, nets_f)

    def train_instance(self, real_A, real_B, z):  # z={0, 1}

        ##### Discriminator loss and optimization
        # NOTE: ".detach()" makes sure no gradient flows to the generator or encoder

        z_lat = self.netZ_2_LAT.forward(z.view(1, 1, 1, 1))
        one_minus_z_lat = self.netZ_2_LAT.forward((1 - z).view(1, 1, 1, 1))

        MA2B = self.netG_A_B.forward(real_A, z_lat)
        MB2A = self.netG_B_A.forward(real_B, one_minus_z_lat)

        ##### Generator and Encoder ALI loss
        # NOTE: The generator and encoder ALI loss is computed using the new (updated)
        # discriminator parameters.

        pred_MA2B_A = self.netD_A.forward(MA2B)
        pred_MA2B_B = self.netD_B.forward(MA2B)
        loss_G_A = (1 - z) * self.criterionGAN(pred_MA2B_A, True) + z * self.criterionGAN(pred_MA2B_B, True)

        pred_MB2A_A = self.netD_A.forward(MB2A)
        pred_MB2A_B = self.netD_B.forward(MB2A)
        loss_G_B = (1 - z) * self.criterionGAN(pred_MB2A_A, True) + z * self.criterionGAN(pred_MB2A_B, True)

        ##### cycle loss
        rec_A = self.netG_B_A.forward(MA2B, z_lat)
        #         print(rec_A.shape, real_A.shape)
        loss_cycle_A = self.criterionCycle(rec_A, real_A)

        rec_B = self.netG_A_B.forward(MB2A, one_minus_z_lat)
        #         print(rec_B.shape, real_B.shape)
        loss_cycle_B = self.criterionCycle(rec_B, real_B)

        ##### Generation optimization
        loss_cycle = loss_cycle_A * self.opt.lambda_A + loss_cycle_B * self.opt.lambda_B
        loss_G = loss_G_A + loss_G_B + loss_cycle

        self.optimizer_Z_2_LAT.zero_grad()
        self.optimizer_G_A.zero_grad()
        self.optimizer_G_B.zero_grad()

        loss_G.backward()

        gnorm_G_A_B = torch.nn.utils.clip_grad_norm_(self.netG_A_B.parameters(), self.opt.max_gnorm)
        gnorm_G_B_A = torch.nn.utils.clip_grad_norm_(self.netG_B_A.parameters(), self.opt.max_gnorm)
        gnorm_Z_2_LAT = torch.nn.utils.clip_grad_norm_(self.netZ_2_LAT.parameters(), self.opt.max_gnorm)

        self.optimizer_G_A.step()
        self.optimizer_G_B.step()

        ##### Discriminator and Z loss

        MA2B_loss_D_fake_A, MA2B_loss_D_true_A, MA2B_pred_fake_A, MA2B_pred_true_A = discriminate(self.netD_A,
                                                                                                  self.criterionGAN,
                                                                                                  MA2B.detach(), real_A)
        MA2B_loss_D_fake_B, MA2B_loss_D_true_B, MA2B_pred_fake_B, MA2B_pred_true_B = discriminate(self.netD_B,
                                                                                                  self.criterionGAN,
                                                                                                  MA2B.detach(), real_B)
        MA2B_loss_D_A = 0.5 * MA2B_loss_D_fake_A + 0.5 * MA2B_loss_D_true_A
        MA2B_loss_D_B = 0.5 * MA2B_loss_D_fake_B + 0.5 * MA2B_loss_D_true_B

        MB2A_loss_D_fake_A, MB2A_loss_D_true_A, MB2A_pred_fake_A, MB2A_pred_true_A = discriminate(self.netD_A,
                                                                                                  self.criterionGAN,
                                                                                                  MB2A.detach(), real_A)
        MB2A_loss_D_fake_B, MB2A_loss_D_true_B, MB2A_pred_fake_B, MB2A_pred_true_B = discriminate(self.netD_B,
                                                                                                  self.criterionGAN,
                                                                                                  MB2A.detach(), real_B)
        MB2A_loss_D_A = 0.5 * MB2A_loss_D_fake_A + 0.5 * MB2A_loss_D_true_A
        MB2A_loss_D_B = 0.5 * MB2A_loss_D_fake_B + 0.5 * MB2A_loss_D_true_B

        loss_D_A = (1 - z) * MA2B_loss_D_A + (1 - z) * MB2A_loss_D_A
        loss_D_B = z * MA2B_loss_D_B + z * MB2A_loss_D_B

        loss_D = loss_D_A + loss_D_B

        # NOTE: after the following snippet, the discriminator parameters will change
        self.optimizer_D_A.zero_grad()
        self.optimizer_D_B.zero_grad()

        loss_D.backward()

        gnorm_D_A = torch.nn.utils.clip_grad_norm_(self.netD_A.parameters(), self.opt.max_gnorm)
        gnorm_D_B = torch.nn.utils.clip_grad_norm_(self.netD_B.parameters(), self.opt.max_gnorm)

        self.optimizer_Z_2_LAT.step()
        self.optimizer_D_A.step()
        self.optimizer_D_B.step()

        ##### Return dicts
        losses = OrderedDict([('D_A', loss_D_A.item()), ('G_A', loss_G_A.item()), ('Cyc_A', loss_cycle_A.item()),
                              ('D_B', loss_D_B.item()), ('G_B', loss_G_B.item()), ('Cyc_B', loss_cycle_B.item()),
                              ('z', z), ])
        visuals = OrderedDict([('real_A', real_A.data), ('MA2B', MA2B.data), ('rec_A', rec_A.data),
                               ('real_B', real_B.data), ('MB2A', MB2A.data), ('rec_B', rec_B.data)])

        if self.opt.monitor_gnorm:
            gnorms = OrderedDict([('gnorm_G_A_B', gnorm_G_A_B),
                                  ('gnorm_G_B_A', gnorm_G_B_A),
                                  ('gnorm_D_B', gnorm_D_B),
                                  ('gnorm_D_A', gnorm_D_A),
                                  ('gnorm_Z_2_LAT', gnorm_Z_2_LAT), ])
            return losses, visuals, gnorms

        return losses, visuals

    def predict_A(self, real_B, z):
        z_lat = self.netZ_2_LAT.forward(z.view(1, 1, 1, 1))
        return self.netG_B_A.forward(real_B, z_lat)

    def predict_B(self, real_A, z):
        z_lat = self.netZ_2_LAT.forward(z.view(1, 1, 1, 1))
        return self.netG_A_B.forward(real_A, z_lat)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_Z_2_LAT.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_B.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def save(self, chk_name):
        chk_path = os.path.join(self.opt.expr_dir, chk_name)
        checkpoint = {
            'netZ_2_LAT': self.netZ_2_LAT.state_dict(),
            'netG_A_B': self.netG_A_B.state_dict(),
            'netG_B_A': self.netG_B_A.state_dict(),
            'netD_A': self.netD_A.state_dict(),
            'netD_B': self.netD_B.state_dict(),
            'optimizer_Z_2_LAT': self.optimizer_Z_2_LAT.state_dict(),
            'optimizer_D_A': self.optimizer_D_A.state_dict(),
            'optimizer_G_A': self.optimizer_G_A.state_dict(),
            'optimizer_D_B': self.optimizer_D_B.state_dict(),
            'optimizer_G_B': self.optimizer_G_B.state_dict()
        }
        torch.save(checkpoint, chk_path)

    def load(self, chk_path):
        checkpoint = torch.load(chk_path)

        self.netZ_2_LAT.load_state_dict(checkpoint['netZ_2_LAT'])
        self.netG_A_B.load_state_dict(checkpoint['netG_A_B'])
        self.netG_B_A.load_state_dict(checkpoint['netG_B_A'])
        self.netD_A.load_state_dict(checkpoint['netD_A'])
        self.netD_B.load_state_dict(checkpoint['netD_B'])
        self.optimizer_Z_2_LAT.load_state_dict(checkpoint['optimizer_Z_2_LAT'])
        self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
        self.optimizer_G_A.load_state_dict(checkpoint['optimizer_G_A'])
        self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
        self.optimizer_G_B.load_state_dict(checkpoint['optimizer_G_B'])

    def eval(self):
        self.netZ_2_LAT.eval()
        self.netG_A_B.eval()
        self.netG_B_A.eval()
        self.netD_A.eval()
        self.netD_B.eval()

    def train(self):
        self.netZ_2_LAT.train()
        self.netG_A_B.train()
        self.netG_B_A.train()
        self.netD_A.train()
        self.netD_B.train()
