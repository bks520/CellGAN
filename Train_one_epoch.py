import torch.optim
import os
import time
from utils import *
import UC_Config as config
import warnings
warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
#=================================================================================
#          训练一轮
#=================================================================================
##################################################################################
def train_one_epoch(loader, model_G,model_D, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger,loss_min,iou_max,dice_max):
    logging_mode = 'Train' if model_G.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0

    dices = []
    for i, (sampled_batch, names) in enumerate(loader,1):
        # print(sampled_batch)
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # 将数据转入GPU
        images, masks = sampled_batch['image'], sampled_batch['label']
        images, masks = images.cuda(), masks.cuda()

        # ====================================================
        #             计算损失
        # ====================================================

        # 损失函数
        d_optimizer = torch.optim.Adam(model_D.parameters(), lr=0.0003)  # 优化器，优化判别器

        #生成器分割的损失
        criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5)
        criterion_D = nn.BCELoss()

        preds = model_G(images)
        preds = preds[:, 0, :, :]
        out_loss_g = criterion(preds.float(), masks.float())  # Loss  换成对数损失函数，
        out_loss=out_loss_g
        print("seg_loss:",out_loss)

        preds=preds.unsqueeze(dim=1)
        print('images.shape:',images.shape,'preds.shape:',preds.shape,'masks.shape',masks.shape)

        fake_img=torch.cat((images,preds),dim=1)  # 拼接原图和生成的图片
        fake_out=model_D(fake_img)  # 判别器的输出
        # print("fake_out:",fake_out[0])
        fake_label = torch.tensor([[0.],[0.],[0.],[0.]])
        print("fake_out, fake_label",fake_out, fake_label)
        d_loss_fake = criterion_D(fake_out, fake_label)   #计算判别器输出和标签的loss
        fake_scores = fake_out #print

        #real的值

        masks_g=torch.unsqueeze(masks,dim=1)
        print("images:",images.shape,"masks",masks.shape)
        real_img = torch.cat((images, masks_g), dim=1)  # 拼接原图和生成的图片
        real_out = model_D(real_img)  # 判别器的输出
        real_label = torch.tensor([[1.],[1.],[1.],[1.]])  # 希望判别器对real_img输出为1 [128,
        d_loss_real = criterion_D(real_out, real_label)  # 计算判别器输出和标签的loss
        real_scores = real_out #print

        d_loss = d_loss_real + d_loss_fake

        out_loss_g=out_loss_g*100+d_loss
        # 分割的loss优化
        if model_G.training:

            d_optimizer.zero_grad()  # 判别器的优化
            d_loss.backward(retain_graph=True)


            optimizer.zero_grad()  #生成器的优化
            print(out_loss_g)
            out_loss_g.backward(retain_graph=True)
            # out_loss.backward()
            d_optimizer.step()
            optimizer.step()


        # train_iou = 0
        train_iou = iou_on_batch(masks,preds)
        train_dice = 0

        batch_time = time.time() - end
        if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        torch.cuda.empty_cache()

    step1 = epoch+1
    writer.add_scalar(logging_mode +'_'+loss_name + '_avg', average_loss.item(), step1)
    if average_loss<loss_min:
        loss_min=average_loss
    writer.add_scalar(logging_mode +'_'+ loss_name + '_min', loss_min.item(), step1)


    # IoU 写入 tensorboard
    writer.add_scalar(logging_mode + '_iou_avg', train_iou_average.item(), step1)
    if train_iou_average>iou_max:
        iou_max=train_iou_average
    writer.add_scalar(logging_mode + '_iou_max', iou_max.item(), step1)
    # writer.add_scalar(logging_mode + '_acc', train_acc, step)


    writer.add_scalar(logging_mode + '_dice_avg', train_dice_avg.item(), step1)
    if train_dice_avg>dice_max:
        dice_max=train_dice_avg
    writer.add_scalar(logging_mode + '_dice_max', dice_max.item(), step1)

    if lr_scheduler is not None:
        lr_scheduler.step()
    # if epoch + 1 > 10: # Plateau
    #     if lr_scheduler is not None:
    #         lr_scheduler.step(train_dice_avg)

    return average_loss, train_dice_avg,loss_min,iou_max,dice_max

