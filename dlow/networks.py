import functools

import torch
import torch.nn as nn
from modules import CondInstanceNorm, TwoInputSequential, CINResnetBlock


###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_Z_2_LAT(input_nc, output_nc, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    netZ_2_LAT = torch.nn.ConvTranspose2d(input_nc, output_nc, kernel_size=1)
    if use_gpu:
        netZ_2_LAT.cuda()
    netZ_2_LAT.apply(weights_init)
    return netZ_2_LAT


def define_stochastic_G(nlatent, input_nc, output_nc, ngf, norm='instance', which_model_netG='resnet',
                        use_dropout=False, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    norm_layer = CondInstanceNorm
    netG = CINResnetGenerator(nlatent, input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              n_blocks=9, gpu_ids=gpu_ids)
    if use_gpu:
        netG.cuda()
    netG.apply(weights_init)
    return netG


def define_D_A(input_nc, ndf, norm, use_sigmoid=False, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    norm_layer = get_norm_layer(norm_type=norm)
    netD = Discriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD


def define_D_B(input_nc, ndf, norm, use_sigmoid=False, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    norm_layer = get_norm_layer(norm_type=norm)
    netD = Discriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD


def print_network(net, out_f=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    if out_f is not None:
        out_f.write(net.__repr__() + "\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()


##############################################################################
# Network Classes
##############################################################################

######################################################################
# Modified version of ResnetGenerator that supports stochastic mappings
# using Conditonal instance norm (can support CBN easily)
######################################################################
class CINResnetGenerator(nn.Module):
    def __init__(self, nlatent, input_nc, output_nc, ngf=64, norm_layer=CondInstanceNorm,
                 use_dropout=False, n_blocks=9, gpu_ids=[], padding_type='zero'):
        assert (n_blocks >= 0)
        super(CINResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        model = [
            torch.nn.ConstantPad2d(3, 0),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2 * ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(2 * ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(2 * ngf, 4 * ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4 * ngf, nlatent),
            nn.ReLU(True)
        ]

        for i in range(3):
            model += [CINResnetBlock(x_dim=4 * ngf, z_dim=nlatent, padding_type=padding_type, norm_layer=norm_layer,
                                     use_dropout=use_dropout, use_bias=True)]

        model += [
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            norm_layer(2 * ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(2 * ngf, ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]

        self.model = TwoInputSequential(*model)

    def forward(self, input, noise):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, noise), self.gpu_ids)
        else:
            return self.model(input, noise)


######################################################################
# Discriminator network
######################################################################
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, gpu_ids=[]):
        """
        input_nc: number of channels in input and output (assumes both inputs are concatenated)
        """
        super(Discriminator, self).__init__()
        self.gpu_ids = gpu_ids

        use_bias = True

        kw = 4
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, 2 * ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(2 * ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(2 * ndf, 4 * ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(4 * ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4 * ndf, 4 * ndf, kernel_size=kw, stride=1, padding=1, bias=use_bias),
            norm_layer(4 * ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4 * ndf, 1, kernel_size=kw, stride=1, padding=1),
            torch.nn.AvgPool2d(kernel_size=(23, 23)),
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids).view(-1, 1)
        else:
            return self.model(input).view(-1, 1)
