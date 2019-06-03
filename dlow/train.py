#!/usr/bin/env python

import glob
import json
import os
import random
import shutil
# !/usr/bin/env python
import sys
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from shutil import copyfile

import numpy as np
import torch
import torchvision.utils as vutils
from model import DLOW
from options import TrainOptions, create_sub_dirs
from sklearn.model_selection import StratifiedShuffleSplit
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset

sys.path.append('..')
import data
import paths
import routine
import functools


def save_results(expr_dir, results_dict):
    # save to results.json (for cluster exp)
    fname = os.path.join(expr_dir, 'results.json')
    with open(fname, 'w') as f:
        json.dump(results_dict, f, indent=4)


def copy_scripts_to_folder(expr_dir):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in glob.glob("%s/*.py" % dir_path):
        shutil.copy(f, expr_dir)


def print_log(out_f, message):
    out_f.write(message + "\n")
    out_f.flush()
    print(message)


def format_log(epoch, i, errors, t, prefix=True):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    if not prefix:
        message = ' ' * len(message)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message


def visualize_cycle(opt, real_A, visuals, eidx, uidx, train):
    size = real_A.size()

    images = [img.cpu().unsqueeze(1) for img in visuals.values()]
    vis_image = torch.cat(images, dim=1).view(size[0] * len(images), size[1], size[2], size[3])
    if train:
        save_path = opt.train_vis_cycle
    else:
        save_path = opt.vis_cycle
    save_path = os.path.join(save_path, 'cycle_%02d_%04d.png' % (eidx, uidx))
    vutils.save_image(vis_image.cpu(), save_path, nrow=len(images))
    copyfile(save_path, os.path.join(opt.vis_latest, 'cycle.png'))


def AbsMaxScale(img, absmax):
    return img / absmax


def get_loader(batch_size):
    la5_dataset = data.LA5_Siblings_MRI(
        paths=['../' + paths.la5_data[0]],
        target_path='../' + paths.la5_target_path,
        load_online=True,
        mri_type="sMRI",
        mri_file_suffix=paths.la5_smri_file_suffix,
        brain_mask_suffix=paths.la5_smri_brain_mask_suffix,
        coord_min=(20, 20, 0),
        img_shape=(153, 189, 163),
        problems=['Schz/Control'],
        temp_storage_path='../' + paths.la5_temp_npy_folder_path
    )
    la5_absmax = 435.2967834472656
    la5_dataset.transform = functools.partial(AbsMaxScale, absmax=la5_absmax)
    sibl_dataset = data.LA5_Siblings_MRI(
        paths=['../' + paths.sibl_data[0]],
        target_path='../' + paths.sibl_target_path,
        load_online=True,
        mri_type="sMRI",
        mri_file_suffix=paths.sibl_smri_file_suffix,
        brain_mask_suffix=paths.sibl_smri_brain_mask_suffix,
        coord_min=(20, 20, 0),
        img_shape=(153, 189, 163),
        problems=['Schz/Control'],
        temp_storage_path='../' + paths.sibl_temp_npy_folder_path,
    )
    sibl_absmax = 730.7531127929688
    sibl_dataset.transform = functools.partial(AbsMaxScale, absmax=sibl_absmax)
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    cv_splits = list(cv.split(X=np.arange(len(sibl_dataset)), y=sibl_dataset.labels))
    train_idx, _ = cv_splits[0]
    train_idx = routine.stratified_batch_indices(train_idx, sibl_dataset.labels[train_idx])
    train_data = data.UnalignedSlicesDataset(la5_dataset, Subset(sibl_dataset, train_idx), la5_dataset.img_shape[:2])
    return DataLoader(train_data, batch_size)


def train_model():
    opt = TrainOptions().parse(sub_dirs=['vis_multi', 'vis_cycle', 'vis_latest', 'train_vis_cycle'])
    out_f = open("%s/results.txt" % opt.expr_dir, 'w')  # сохранить скрипты на случай изменений
    copy_scripts_to_folder(opt.expr_dir)  # сохранить скрипты на случай изменений
    use_gpu = len(opt.gpu_ids) > 0

    # reproducibility
    if opt.seed is not None:
        print("using random seed:", opt.seed)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        if use_gpu:
            torch.cuda.manual_seed_all(opt.seed)

    train_data_loader = get_loader(opt.batchSize)
    model = DLOW(opt)
    print_log(out_f, "model [%s] was created" % (model.__class__.__name__))

    # visualizer = Visualizer(opt)
    total_steps = 0
    print_start_time = time.time()
    create_sub_dirs(opt, ['vis_pred_B'])

    model.train()
    T = (opt.niter + opt.niter_decay + 1) * len(train_data_loader)  # for generating z
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_data_loader):
            real_A, real_B = Variable(data['A']), Variable(data['B'])
            if real_A.size(0) != real_B.size(0):
                continue

            t = epoch * len(train_data_loader) + i
            alpha = np.exp((t - 0.5 * T) / (0.25 * T))
            z = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([1.])).sample()

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            if use_gpu:
                real_A = real_A.cuda()
                real_B = real_B.cuda()
                z = z.cuda()

            if opt.monitor_gnorm:
                losses, visuals, gnorms = model.train_instance(real_A, real_B, z)
            else:
                losses, visuals = model.train_instance(real_A, real_B, z)

            if total_steps % opt.display_freq == 0:
                # visualize current training batch
                visualize_cycle(opt, real_A, visuals, epoch, epoch_iter / opt.batchSize, train=True)

            if total_steps % opt.print_freq == 0:
                t = (time.time() - print_start_time) / opt.batchSize
                print_log(out_f, format_log(epoch, epoch_iter, losses, t))
                if opt.monitor_gnorm:
                    print_log(out_f, format_log(epoch, epoch_iter, gnorms, t, prefix=False) + "\n")
                print_start_time = time.time()
            del real_A, real_B

        if epoch % opt.save_epoch_freq == 0:
            print_log(out_f, 'saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')

        print_log(out_f, 'End of epoch %d / %d \t Time Taken: %d sec\n' % (
        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()

    out_f.close()


if __name__ == "__main__":
    train_model()
