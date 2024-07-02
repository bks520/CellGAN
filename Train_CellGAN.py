import argparse
import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random

from torch import nn
from torch.backends import cudnn
from optparse import OptionParser

from discriminator_model import Discriminator
from Load_Dataset import RandomGenerator,ValGenerator,ImageToImage2D
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch
import UC_Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE
from vit_seg_modeling import CONFIGS
from vit_seg_modeling import VisionTransformer


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)

def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)

##################################################################################
#=================================================================================
#          加载模型，训练
#=================================================================================
##################################################################################

def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # 加载数据
    train_tf= transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    train_dataset = ImageToImage2D(config.train_dataset, train_tf,image_size=config.img_size)
    val_dataset = ImageToImage2D(config.val_dataset, val_tf,image_size=config.img_size)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)

    lr = config.learning_rate
    logger.info(model_type)

    parser = OptionParser()

    def get_comma_separated_int_args(option, opt, value, parser):
        value_list = value.split(',')
        value_list = [int(i) for i in value_list]
        setattr(parser.values, option.dest, value_list)

    config_vit1 = CONFIGS['R50-ViT-B_16']
    config_vit1.n_classes = 1
    config_vit1.n_skip = 3
    config_vit1.patches.grid = (int(224 / 16), int(224 / 16))

    if model_type == 'CellGAN':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        # 生成器加载
        model_G =VisionTransformer(config_vit1, img_size=224, num_classes=config_vit1.n_classes).cuda()
        # 判别器加载
        model_D=Discriminator(in_channels=4).cuda() # 创建判别器

    #计算损失
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_G.parameters()), lr=lr)  # Choose optimize
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler =  None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    loss_min = 99
    iou_max = 0
    dice_max = 0
    loss_min1 = 99
    iou_max1 = 0
    dice_max1 = 0
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs ))
        logger.info(config.session_name)
        # 模型训练
        model_G.train(True)
        model_D.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        A,B,C,D,E=train_one_epoch(train_loader, model_G,model_D, criterion, optimizer, writer, epoch, None, model_type, logger,loss_min,iou_max,dice_max)
        loss_min, iou_max, dice_max=C,D,E
        # 在验证集上测试
        logger.info('Validation')
        with torch.no_grad():
            model_G.eval()
            model_D.eval()
            val_loss, val_dice, C1,D1,E1= train_one_epoch(val_loader, model_G,model_D, criterion,
                                            optimizer, writer, epoch, lr_scheduler,model_type,logger,loss_min1,iou_max1,dice_max1)
        loss_min1, iou_max1, dice_max1=C1,D1,E1
        # =============================================================
        #       保存在验证集上的best model
        # =============================================================
        if val_dice > max_dice:
            if epoch+1 > 5:
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice,max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count,config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            #break

    return model


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)

