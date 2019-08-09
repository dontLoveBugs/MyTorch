from __future__ import division
from tqdm import tqdm

import torch
import torch.nn as nn

import sys

sys.path.append("..")

from modules.utils.config import Config

from run.data import get_train_loader
from run.network import PSPNet
from run.validator import Validator

from modules.datasets.seg.ade import ADE
from modules.utils.init_func import init_weight, group_weight
from modules.utils.pyt_utils import all_reduce_tensor
from modules.utils.average_meter import AverageMeter
from modules.engine.lr_policy import PolyLR
from modules.engine.engine import Engine

try:
    from apex.parallel import SyncBatchNorm, DistributedDataParallel
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

"""
Usage: python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
"""

# read trainig confi file
config = Config(config_file='./config.json')

with Engine(config=config) as engine:
    loss_meter = AverageMeter()
    # set validator to monitor training engine.
    if engine.local_rank == 0:
        # save config
        engine.save_config(config.log.snapshot_dir)
        val_set = ADE(config.data.dataset_path, split='val')
        validator = Validator(dataset=val_set,
                              device=engine.local_rank,
                              config=config, out_id=[1])

    # data loader
    train_loader, train_sampler, niters_per_epoch = get_train_loader(engine, ADE)
    niters_per_epoch = 100

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=-1)

    if engine.distributed:
        engine.logger.info('Use the Multi-Process-SyncBatchNorm')
        BatchNorm2d = SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    model = PSPNet(config.data.num_classes, criterion=criterion,
                   pretrained_model=config.model.pretrained_model,
                   norm_layer=BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.model.bn_eps, config.model.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # group weight and config optimizer
    base_lr = config.train.lr

    params_list = []
    params_list = group_weight(params_list, model.backbone,
                               BatchNorm2d, base_lr)
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d,
                                   base_lr * 10)

    # config lr policy
    total_iteration = config.train.nepochs * niters_per_epoch
    lr_policy = PolyLR(base_lr, config.train.lr_power, total_iteration)
    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.train.momentum,
                                weight_decay=config.train.weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if engine.distributed:
        model = DistributedDataParallel(model)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)

    # read stored model.
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()

    for epoch in range(engine.state.epoch, config.train.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            loss = model(imgs, gts)

            # reduce the whole loss over multi-gpu
            reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size) if engine.distributed else loss
            loss_meter.update(reduce_loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # reset learning rate
            current_idx = epoch * niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

            print_str = 'Epoch{}/{}'.format(epoch, config.train.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % reduce_loss.item() \
                        + '(%.2f)' % loss_meter.avg

            pbar.set_description(print_str, refresh=False)

        if engine.distributed and (engine.local_rank == 0):
            engine.save_and_link_checkpoint(config.log.snapshot_dir,
                                            config.log.log_dir,
                                            config.log.log_dir_link)
        elif not engine.distributed:
            engine.save_and_link_checkpoint(config.log.log.snapshot_dir,
                                            config.log.log_dir,
                                            config.log.log_dir_link)

        # visualization
        if engine.local_rank == 0:
            val_loss, result, out_imgs = validator.eval(model.module)
            engine.tb_logger.add_scalars_list('TrainVal', [{'train': loss_meter.avg, 'val': val_loss}], epoch)
            engine.tb_logger.add_scalar_dicts('Train', result, epoch)
            loss_meter.reset()

            engine.save_images(config.log.snapshot_dir,
                               'compare_' + str(epoch) + '.png', out_imgs)
            engine.logger.info('Epoch {} Validation: '.format(epoch)
                               + 'mean acc:%.2f' % result.acc
                               + ', mean iou:%.2f' % result.mean_iu)

        model.train()
