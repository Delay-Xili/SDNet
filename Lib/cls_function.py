# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import torch

logger = logging.getLogger(__name__)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        if len(topk) == 1:
            res.append(torch.zeros_like(res[-1]))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict, topk=(1, 5)):
    batch_time = AverageMeter()
    model_time = AverageMeter()
    losses = AverageMeter()
    loss_cls = AverageMeter()
    loss_r = AverageMeter()
    loss_c = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    effec_batch_num = len(train_loader)
    # effec_batch_num = int(config.PERCENT * total_batch_num)
    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        # compute output
        # end = time.time()
        model_time.update(time.time() - start)

        model.module.update_stepsize() if isinstance(model, torch.nn.DataParallel) else model.update_stepsize()
        output, extra = model(input.cuda())

        target = target.cuda(non_blocking=True)
        if torch.cuda.device_count() > 1 and extra is not None:
            extra = process_for_parallel(extra)

        loss = criterion(output, target)
        # record cls loss
        loss_cls.update(loss.item(), input.size(0))

        if extra is not None and config.MODEL.DICTLOSS:
            loss_r.update(extra[0].item(), input.size(0))
            loss_c.update(extra[1].item(), input.size(0))
            loss = loss + config.MODEL.RCLOSS_FACTOR * (extra[0] + extra[1])

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()

        # measure accuracy and record total loss
        losses.update(loss.item(), input.size(0))

        prec1, prec5 = accuracy(output, target, topk=topk)

        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Speed {speedv:.1f} ({speeda:.1f}) samples/s\t' \
                  'Data ({data.avg:.2f}s)\t' \
                  'Total_Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t' \
                  'Loss_r ({loss_r.avg:.4f})\t'\
                  'Loss_c ({loss_c.avg:.4f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f}\t'.format(
                      epoch, i, effec_batch_num,
                      speedv=input.size(0)/batch_time.val,
                      speeda=input.size(0)/batch_time.avg,
                      data=model_time,
                      loss=losses, loss_cls=loss_cls, loss_r=loss_r, loss_c=loss_c,
                      top1=top1, top5=top5
            )
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer.add_scalar('train_top1', top1.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def validate(config, val_loader, model, criterion, lr_scheduler, epoch, output_dir, tb_log_dir,
             writer_dict=None, topk=(1, 5)):
    batch_time = AverageMeter()
    losses = AverageMeter()
    loss_cls = AverageMeter()
    loss_r = AverageMeter()
    loss_c = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # compute output

            if isinstance(model, torch.nn.DataParallel):
                model.module.update_stepsize()
            else:
                model.update_stepsize()
            output, extra = model(input)
            target = target.cuda()
            if torch.cuda.device_count() > 1 and extra is not None:
                extra = process_for_parallel(extra)

            loss = criterion(output, target)
            # record cls loss
            loss_cls.update(loss.item(), input.size(0))
            if extra is not None and config.MODEL.DICTLOSS:
                loss_r.update(extra[0].item(), input.size(0))
                loss_c.update(extra[1].item(), input.size(0))
                loss = loss + config.MODEL.RCLOSS_FACTOR * (extra[0] + extra[1])

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output, target, topk=topk)
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Loss_cls {loss_cls.avg:.4f}\t' \
              'Loss_r {loss_r.avg:.4f}\t' \
              'Loss_c {loss_c.avg:.4f}\t' \
              'Error@1 {error1:.3f}\t' \
              'Error@5 {error5:.3f}\t' \
              'Accuracy@1 {top1.avg:.3f}\t' \
              'Accuracy@5 {top5.avg:.3f}\t'.format(
                  batch_time=batch_time,
                  loss=losses, loss_cls=loss_cls, loss_r=loss_r, loss_c=loss_c,
                  top1=top1, top5=top5,
                  error1=100-top1.avg, error5=100-top5.avg
        )
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_top1', top1.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return (top1.avg, losses.avg, loss_cls.avg, loss_r.avg, loss_c.avg)


def process_for_parallel(extra):

    out = [item.sum() for item in extra[:-1]]
    last = [item.sum() for item in extra[-1]]
    out.append(last)
    return out
