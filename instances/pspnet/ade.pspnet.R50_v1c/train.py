from __future__ import division
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from modules.utils.config import Config

from .dataloader import get_train_loader
from .network import PSPNet

from modules.datasets.seg.ade import ADE
from modules.utils.init_func import init_weight, group_weight
from modules.utils.pyt_utils import all_reduce_tensor
from modules.engine.lr_policy import PolyLR
from modules.engine.logger import get_logger
from modules.engine.engine import Engine

try:
    from third_libs.parallel import SyncBatchNorm, DistributedDataParallel
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

logger = get_logger()

# 读取配置文件
config = Config(config_file='./config.json').get_config()

torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

parser = argparse.ArgumentParser()

with Engine(config=config) as engine:
    # args = parser.parse_args()

    cudnn.benchmark = True
    if engine.distributed:
        torch.cuda.set_device(engine.local_rank)

    # data loader
    train_loader, train_sampler, niters_per_epoch = get_train_loader(engine, ADE)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=-1)

    if engine.distributed:
        logger.info('Use the Multi-Process-SyncBatchNorm')
        BatchNorm2d = SyncBatchNorm

    model = PSPNet(config, criterion=criterion,
                   pretrained_model=config.pretrained_model,
                   norm_layer=BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
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
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(engine.niters_per_epoch), file=sys.stdout,
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
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss,
                                                world_size=engine.world_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = epoch * engine.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % reduce_loss.item()

            pbar.set_description(print_str, refresh=False)

        if (epoch >= config.nepochs - 20) or (
                epoch % config.snapshot_iter == 0):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
