import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
from skimage import measure
from typing import Optional

def SoftIoULoss( pred, target):
        pred = torch.sigmoid(pred)
  
        smooth = 1

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        loss = (intersection_sum + smooth) / \
                    (pred_sum + target_sum - intersection_sum + smooth)
    
        loss = 1 - loss.mean()

        return loss

def Dice( pred, target,warm_epoch=1, epoch=1, layer=0):
        pred = torch.sigmoid(pred)
  
        smooth = 1

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))

        loss = (2*intersection_sum + smooth) / \
            (pred_sum + target_sum + intersection_sum + smooth)

        loss = 1 - loss.mean()

        return loss

class SLSIoULoss(nn.Module):
    """
    修改后的 SLSIoULoss，适配目标项目的调用方式：
    loss += self.loss_fn(pred_logit.sigmoid(), batch_masks)
    
    它现在接收概率图 (pred)，内部处理 epoch 和 warm_epoch，并进行标签降采样。
    """
    def __init__(self, warm_epoch: int = 1, get_epoch_fn: Optional[callable] = None):
        super(SLSIoULoss, self).__init__()
        # 存储 warm_epoch
        self.warm_epoch = warm_epoch
        # 存储一个用于获取当前 epoch 的函数 (例如：lambda: trainer.epoch)
        self.get_epoch = get_epoch_fn 
        # 检查是否设置了 epoch 获取函数
        if self.get_epoch is None:
             print("Warning: SLSIoULoss is running without a provided get_epoch_fn. Shape loss will not be controlled by epoch.")


    # 注意：forward方法的签名必须匹配目标训练代码的调用：
    # loss += self.loss_fn(pred_logit.sigmoid(), batch_masks)
    def forward(self, pred: torch.Tensor, target_full_res: torch.Tensor, with_shape: bool = True):
        # 1. 获取当前 epoch
        current_epoch = self.get_epoch() if self.get_epoch else (self.warm_epoch + 1)
        
        # 2. **关键修改：移除 torch.sigmoid(pred)**
        #    因为训练代码已经传入了 pred_logit.sigmoid()，pred 已经是概率图 [0, 1]
        #    如果 pred 是 logits，则需要保留 sigmoid。我们假设 pred 是概率图。
        #    为了保持原 SLSIoULoss 的变量名，这里将 pred 视为概率图。
        
        # 3. **多尺度标签降采样**
        #    target_full_res 是原始的大尺寸标签 (batch_masks)
        #    pred 是当前尺度的预测概率图
        H_pred, W_pred = pred.shape[2], pred.shape[3]
        H_target, W_target = target_full_res.shape[2], target_full_res.shape[3]
        
        target = target_full_res
        if H_pred != H_target or W_pred != W_target:
            # 标签降采样以匹配当前预测尺寸
            target = F.interpolate(
                target_full_res.float(), 
                size=(H_pred, W_pred), 
                mode='nearest'
            ).long().to(target_full_res.device)
        
        # 4. **核心 IoU 和权重计算 (基于原始 SLSIoULoss 逻辑)**
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        dis = torch.pow((pred_sum-target_sum)/2, 2)
        
        # alpha: 面积相似度权重
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / \
                (torch.max(pred_sum, target_sum) + dis + smooth) 
        
        # IoU 值 (非损失)
        loss_val = (intersection_sum + smooth) / \
                   (pred_sum + target_sum - intersection_sum  + smooth)
        
        # beta: 样本权重
        beta = torch.sum(target, dim = (1, 2, 3)) / (torch.sum(target) + np.spacing(1))
        
        # 形状损失
        lloss = LLoss(pred, target)
        
        # 5. **分阶段损失计算**
        if current_epoch > self.warm_epoch:       
            siou_loss_val = alpha * loss_val # 使用 alpha 调整 IoU 值
            if with_shape:
                loss = (beta * (1 - siou_loss_val)).sum() + lloss # 损失 = (1 - 调整后IoU) + 形状损失
            else:
                loss = (beta * (1 - siou_loss_val)).sum()
        else:
            # 预热期：只使用基础加权的 IoU 损失
            loss = (beta * (1 - loss_val)).sum()
            
        return loss
    
def LLoss(pred, target):
        loss = torch.tensor(0.0, requires_grad=True).to(pred)

        patch_size = pred.shape[0]
        h = pred.shape[2]
        w = pred.shape[3]        
        x_index = torch.arange(0,w,1).view(1, 1, w).repeat((1,h,1)).to(pred) / w
        y_index = torch.arange(0,h,1).view(1, h, 1).repeat((1,1,w)).to(pred) / h
        smooth = 1e-8
        for i in range(patch_size):  

            pred_centerx = (x_index*pred[i]).mean()
            pred_centery = (y_index*pred[i]).mean()

            target_centerx = (x_index*target[i]).mean()
            target_centery = (y_index*target[i]).mean()
           
            angle_loss = (4 / (torch.pi**2) ) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth)) 
                                                             - torch.arctan((target_centery) / (target_centerx + smooth))))
            pred_length = torch.sqrt(pred_centerx*pred_centerx + pred_centery*pred_centery + smooth)
            target_length = torch.sqrt(target_centerx*target_centerx + target_centery*target_centery + smooth)

            length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)

            loss = loss + (1 - length_loss + angle_loss) / patch_size
        
        return loss

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
