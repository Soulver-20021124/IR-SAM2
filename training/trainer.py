"""
Trainer Module for SAM-SPL Model Training

This module provides a comprehensive training framework for the SAM-SPL model.
It supports both single-GPU and distributed training, with features for
training, validation, checkpointing, and metric evaluation.

Key Features:
- Distributed training support with PyTorch DDP
- Progress tracking with enlighten progress bars
- Automatic checkpoint saving and loading
- Flexible metric evaluation system
- Support for custom loss functions
"""

import os
from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from loguru import logger
import enlighten
import cv2
import random
import numpy as np
from tqdm import tqdm # 用于显示预计算进度

# Import metricWrapper for type hints
from .metrics_config import metricWrapper

def get_mIoU(metrics_wrapper):
    """Helper function to extract mIoU from metricWrapper"""
    # 假设 metricWrapper 具有 miou_meter 属性，且其 get() 返回一个包含 mIoU 的元组
    _, mIoU, _ = metrics_wrapper.miou_meter.get()
    return mIoU



class CSALoss(nn.Module):
  
    def __init__(self, 
                 warm_epoch: int = 1, 
                 get_epoch_fn: Optional[callable] = None,
                 s_mean: float = 49.8790,   
                 c_mean: float = 1.4168,   
                 tda_weight: float = 0.2 
                 ):
        super(CSALoss, self).__init__()
        self.warm_epoch = warm_epoch
        self.get_epoch = get_epoch_fn
        
        # TDA Loss 参数
        self.s_mean = s_mean
        self.c_mean = c_mean
        self.tda_weight = tda_weight

        if self.get_epoch is None:
             print("Warning: Loss is running without get_epoch_fn.")

    def forward(self, pred: torch.Tensor, target_full_res: torch.Tensor, input_images: torch.Tensor = None, activate_shape_loss: bool = True):
        """
        Args:
            pred: 预测概率图 (Batch, 1, H, W)，已经过 sigmoid
            target_full_res: 标签 (Batch, 1, H, W)
            input_images: 原始红外图像 (Batch, 1, H, W), 用于计算 TDA Loss 中的对比度。
                          如果为 None，将退化为只使用尺寸自适应或跳过 TDA。
            activate_shape_loss: 是否激活 SLS 中的形状损失
        """
        # 1. 获取当前 epoch
        current_epoch = self.get_epoch() if self.get_epoch else (self.warm_epoch + 1)
        
        # ---------------------------
        # Part A: 原有 SLS Loss 计算
        # ---------------------------
        H_pred, W_pred = pred.shape[2], pred.shape[3]
        H_target, W_target = target_full_res.shape[2], target_full_res.shape[3]
        
        target = target_full_res
        if H_pred != H_target or W_pred != W_target:
            target = F.interpolate(
                target_full_res.float(), 
                size=(H_pred, W_pred), 
                mode='nearest'
            ).long().to(target_full_res.device)
        
        smooth = 1e-8
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        dis = torch.pow((pred_sum-target_sum)/2, 2)
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / \
                (torch.max(pred_sum, target_sum) + dis + smooth) 
        
        loss_val = (intersection_sum + smooth) / \
                   (pred_sum + target_sum - intersection_sum  + smooth)
        
        #beta = torch.sum(target, dim = (1, 2, 3)) / (torch.sum(target) + np.spacing(1))
        
        lloss = LLoss(pred, target) # 保持你原有的辅助函数不变
        
        # 计算 SLS 基础 Loss
        if current_epoch > self.warm_epoch:       
            siou_loss_val = alpha * loss_val 
            if activate_shape_loss:
                #loss_sls = (beta * (1 - siou_loss_val)).sum() + lloss
                loss_sls = (1 - siou_loss_val).mean() + lloss
            else:
                #loss_sls = (beta * (1 - siou_loss_val)).sum()
                loss_sls = (1 - siou_loss_val).mean()
        else:
            #loss_sls = (beta * (1 - loss_val)).sum()
            loss_sls = (1 - loss_val).mean()

        # ---------------------------
        # Part B: TDA Loss (新加部分)
        # ---------------------------
        # 只有在传入了原图，且当前已经是正式训练阶段(可选)时才计算 TDA
        loss_tda = torch.tensor(0.0, device=pred.device)
        
        # 论文中 TDA 是为了增强对难样本的挖掘，通常全阶段或稳定后使用
        # 这里假设只要有原图就计算
        if input_images is not None and activate_shape_loss:
            # 确保原图尺寸和标签一致 (如果不一致需要缩放原图，通常是一致的)
            if input_images.shape[2:] != target_full_res.shape[2:]:
                 imgs_for_tda = F.interpolate(input_images, size=target_full_res.shape[2:], mode='bilinear', align_corners=False)
            else:
                 imgs_for_tda = input_images

            loss_tda = self.calculate_tda_loss(pred, target, imgs_for_tda)
        
        # ---------------------------
        # Part C: 总损失组合 [cite: 112]
        # ---------------------------
        # L_total = L_S + w_T * L_T
        total_loss = loss_sls + self.tda_weight * loss_tda
        
        return total_loss

    def calculate_tda_loss(self, pred, target, images):
        """
        实现 TDA Loss 的核心逻辑 [cite: 98, 99, 100, 101]
        """
        batch_size = pred.shape[0]
        tda_loss_sum = 0.0
        valid_targets = 0

        # 由于需要提取连通域和裁剪，这部分逻辑在 CPU 上用 OpenCV 处理比较方便
        # 注意：虽然这会引入 GPU-CPU 同步开销，但在小目标检测中目标稀疏，开销通常可接受
        pred_detached = pred.detach().cpu().numpy()
        target_cpu = target.cpu().numpy().astype(np.uint8)
        images_cpu = images.detach().cpu().numpy() # 只需要原图数值计算对比度，不需要梯度
        
        patch_size = 48 # 论文设定的固定 Patch 大小 [cite: 89]

        for b in range(batch_size):
            # 获取当前样本的 Mask 和 Image
            mask_b = target_cpu[b, 0]
            img_b = images_cpu[b, 0]
            
            # 1. 连通域分析 (Spaghetti labeling 或者简单的 connectedComponents) [cite: 87]
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_b, connectivity=8)
            
            if num_labels <= 1: # 只有背景，没有目标
                continue

            # 遍历每个目标 (label 0 是背景，从 1 开始)
            loss_t_accum = 0.0
            targets_in_image = 0
            
            for i in range(1, num_labels):
                # 获取目标统计信息
                x, y, w, h, area = stats[i]
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                
                # 2. 随机膨胀 bounding box (d between 2 and 5) [cite: 88]
                d = random.randint(2, 5)
                x1 = max(0, x - d)
                y1 = max(0, y - d)
                x2 = min(mask_b.shape[1], x + w + d)
                y2 = min(mask_b.shape[0], y + h + d)
                
                # 3. 计算自适应权重 p_t [cite: 98]
                # s_t: 目标像素数 [cite: 104]
                s_t = area 
                
                # c_t: 局部对比度 [cite: 105, 106]
                # 目标区域 mask
                target_pixels_mask = (labels[y1:y2, x1:x2] == i)
                # 背景区域 mask (Patch 内非目标区域)
                bg_pixels_mask = (labels[y1:y2, x1:x2] != i)
                
                patch_img = img_b[y1:y2, x1:x2]
                
                if np.sum(target_pixels_mask) > 0 and np.sum(bg_pixels_mask) > 0:
                    mean_target = np.mean(patch_img[target_pixels_mask])
                    mean_bg = np.mean(patch_img[bg_pixels_mask])
                    c_t = abs(mean_target - mean_bg)
                else:
                    c_t = self.c_mean # 边界情况处理
                
                # 公式 (4): p_t = 1 + sigmoid(-s_t/s_mean) + sigmoid(-c_t/c_mean)
                # 注意：numpy 的 sigmoid 实现
                term_s = 1 / (1 + np.exp(s_t / self.s_mean)) # sigmoid(-x) = 1/(1+exp(x))
                term_c = 1 / (1 + np.exp(c_t / self.c_mean))
                p_t = 1 + term_s + term_c
                
                # 4. 裁剪并缩放预测图和 GT 到 48x48 [cite: 89]
                # 这里需要回到 Tensor 进行操作以保持梯度
                # 坐标归一化到 [-1, 1] 用于 grid_sample 或者直接切片
                # 简单起见直接切片再 interpolate
                
                pred_patch = pred[b:b+1, :, y1:y2, x1:x2]
                target_patch = target[b:b+1, :, y1:y2, x1:x2] # 注意：此时 target 是 float 用于计算
                
                if pred_patch.numel() == 0: continue

                # 缩放到 48x48
                pred_patch_resized = F.interpolate(pred_patch, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                target_patch_resized = F.interpolate(target_patch.float(), size=(patch_size, patch_size), mode='nearest')
                
                # 5. 计算 Soft IoU (I_t) [cite: 98]
                inter = (pred_patch_resized * target_patch_resized).sum()
                union = (pred_patch_resized + target_patch_resized).sum() - inter
                I_t = (inter + 1e-6) / (union + 1e-6)
                
                # 6. 计算单个目标损失 L_t [cite: 98]
                # L_t = -(1 - I_t^p_t) * log(I_t)
                # p_t 是标量，不参与梯度回传，只作为权重
                p_t_tensor = torch.tensor(p_t, device=pred.device)
                
                # 增加 eps 防止 log(0)
                I_t_clamped = torch.clamp(I_t, min=1e-6, max=1.0)
                l_t = -(1 - torch.pow(I_t_clamped, p_t_tensor)) * torch.log(I_t_clamped)
                
                loss_t_accum += l_t
                targets_in_image += 1
            
            if targets_in_image > 0:
                # 公式 (1): L_T = sum(L_t) / N
                tda_loss_sum += (loss_t_accum / targets_in_image)
                valid_targets += 1
        
        if valid_targets > 0:
            return tda_loss_sum / valid_targets
        else:
            return torch.tensor(0.0, device=pred.device)

# 保持原有的 LLoss 不变
def LLoss(pred, target):
    loss = torch.tensor(0.0, requires_grad=True).to(pred.device) # fix .to(pred) to .to(pred.device)

    patch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]        
    x_index = torch.arange(0,w,1).view(1, 1, w).repeat((1,h,1)).to(pred.device) / w
    y_index = torch.arange(0,h,1).view(1, h, 1).repeat((1,1,w)).to(pred.device) / h
    smooth = 1e-8
    
    # 向量化优化：避免在 patch_size 上使用 python for 循环，可以利用 batch 维度计算
    # 但为了不破坏你原有逻辑，暂时保持循环，仅修复 device 问题
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

class Trainer:
    """
    A comprehensive training class for SAM-SPL models.
    
    This class handles the complete training lifecycle including:
    - Model training and validation
    - Distributed training setup
    - Checkpoint management
    - Progress monitoring
    - Metric evaluation
    
    Attributes:
        model: The neural network model to train
        optimizer: Optimizer for model parameters
        scheduler: Learning rate scheduler (optional)
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        loss_fn: Custom loss function (default: BCELoss)
        device: Device to run training on (default: "cuda")
        batch_size: Batch size for training (default: 8)
        num_workers: Number of data loading workers (default: 4)
        distributed: Enable distributed training (default: False)
        save_dir: Directory to save checkpoints (default: "./checkpoints")
        etric_wrapper: Optional metric computation wrapper
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataset: torch.utils.data.Dataset,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        loss_type: str = "bceloss", # <--- 修改1: 接收 loss_type 字符串
        s_mean: float = 50.2567,  # <--- 新增
        c_mean: float = 63.1712,  # <--- 新增
        loss_fn: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 8,
        warm_epoch: int = 20,
        num_workers: int = 4,
        distributed: bool = False,
        save_dir: str = "./checkpoints",
        metric_wrapper: Optional[metricWrapper] = metricWrapper(),
    ):
        self.model = model  # type: nn.Module
        self.optimizer = optimizer  # type: torch.optim.Optimizer
        self.scheduler = scheduler  # type: Optional[torch.optim.lr_scheduler.LRScheduler]
        self.train_dataset = train_dataset # 保存 dataset 引用以便统计
        
        self.device = device  # type: Union[str, torch.device]
        self.save_dir = save_dir  # type: str
        self.metric_wrapper = metric_wrapper  # type: Optional[metricWrapper]
        os.makedirs(self.save_dir, exist_ok=True)
        self.distributed = distributed  # type: bool
        self.rank = dist.get_rank() if distributed else 0  # type: int
        self.world_size = dist.get_world_size() if distributed else 1  # type: int
        self.epoch = 0  # type: int
        self.warm_epoch = 20
        # >>> 新增状态变量：控制形状损失是否激活的状态
        self.activate_shape_loss = False # type: bool
        # >>> 新增：形状损失激活的MiOU阈值 (40%)
        self.shape_loss_miou_threshold = 0.40 # type: float
        # >>> 新增状态变量：追踪是否已经给出形状损失（Shape Loss）激活的提示
        self.warned_shape_loss = False # type: bool
        # --- 修改2: 根据 loss_type 初始化不同的 loss_fn ---
        self.loss_type = loss_type.lower()
        if self.loss_type == 'csaloss':
            if self.rank == 0:
                logger.info("⏳ Calculating dataset statistics for CSALoss (s_mean, c_mean)...")
                # 调用内部统计方法
            s_mean, c_mean = self._calculate_dataset_statistics(train_dataset)
            if self.rank == 0:
                logger.info(f"✅ Statistics Done: s_mean={s_mean:.4f}, c_mean={c_mean:.4f}")
                logger.info(f"Trainer initialized with CSALoss (Warm-up: {self.warm_epoch} epochs)")
            self.loss_fn = CSALoss(
                warm_epoch=self.warm_epoch, 
                get_epoch_fn=lambda: self.epoch,
                s_mean=s_mean,  # <--- 传入
                c_mean=c_mean,  # <--- 传入
                tda_weight=0.2,  # 论文推荐值 [cite: 115]
            )
        elif self.loss_type == 'bceloss':
            if self.rank == 0:
                logger.info("Trainer initialized with BCELoss")
            # 使用标准的 BCELoss
            self.loss_fn = nn.BCELoss(reduction="mean")
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        self.best_metric = None  # type: Optional[Any]
        self.train_sampler = (
            DistributedSampler(train_dataset) if distributed else None
        )  # type: Optional[DistributedSampler]
        self.val_sampler = (
            DistributedSampler(val_dataset)
            if (distributed and val_dataset is not None)
            else None
        )  # type: Optional[DistributedSampler]
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
        )  # type: DataLoader
        self.val_loader = None  # type: Optional[DataLoader]
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                sampler=self.val_sampler,
            )  # type: DataLoader
        self.model = self.model.to(self.device)
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device] if torch.cuda.is_available() else None,
                output_device=self.rank,
                find_unused_parameters=True,
            )

    def _calculate_dataset_statistics(self, dataset) -> Tuple[float, float]:
        """
        内部方法：遍历 Dataset 采样子集计算 s_mean 和 c_mean。
        为了效率，不使用 DataLoader，直接遍历 dataset 索引。
        """
        all_sizes = []
        all_contrasts = []
        
        # 为了不拖慢 DDP 启动，如果你想只在 rank0 计算然后广播也可以，
        # 但考虑到数据集较小，所有 rank 独立计算最简单且不容易出错。
        
        # 使用 tqdm 显示进度 (只在 rank 0 显示)
        iterator = range(len(dataset))
        if self.rank == 0:
            iterator = tqdm(iterator, desc="Scanning Dataset", unit="img")
            
        for i in iterator:
            try:
                # 获取样本：假设 dataset[i] 返回 (image, mask, ...)
                # image: tensor (C, H, W)
                # mask: tensor (1, H, W) 或 (H, W)
                sample = dataset[i] 
                img_tensor = sample[0]
                mask_tensor = sample[1]
                
                # 转换为 Numpy 用于 OpenCV 计算
                # 注意：img_tensor 可能是归一化过的 (float)，这没问题，只要训练时也用归一化的图即可
                if torch.is_tensor(img_tensor):
                    img_np = img_tensor.detach().cpu().numpy()
                    if img_np.ndim == 3: img_np = img_np[0] # 取单通道
                else:
                    img_np = np.array(img_tensor)
                    
                if torch.is_tensor(mask_tensor):
                    mask_np = mask_tensor.detach().cpu().numpy()
                    if mask_np.ndim == 3: mask_np = mask_np[0]
                else:
                    mask_np = np.array(mask_tensor)
                
                # 确保 mask 是 uint8 二值图
                mask_uint8 = (mask_np > 0).astype(np.uint8) * 255
                
                # --- 连通域分析 ---
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                
                # 跳过背景 (label 0)
                for label_idx in range(1, num_labels):
                    # [cite_start]1. 收集尺寸 (s_t) [cite: 104]
                    area = stats[label_idx, cv2.CC_STAT_AREA]
                    all_sizes.append(area)
                    
                    # [cite_start]2. 收集对比度 (c_t) [cite: 105, 106]
                    # 模拟 Patch 机制
                    x = stats[label_idx, cv2.CC_STAT_LEFT]
                    y = stats[label_idx, cv2.CC_STAT_TOP]
                    w = stats[label_idx, cv2.CC_STAT_WIDTH]
                    h = stats[label_idx, cv2.CC_STAT_HEIGHT]
                    
                    # 膨胀 3 像素 (模拟论文平均值)
                    d = 3 
                    x1 = max(0, x - d)
                    y1 = max(0, y - d)
                    x2 = min(img_np.shape[1], x + w + d)
                    y2 = min(img_np.shape[0], y + h + d)
                    
                    patch_img = img_np[y1:y2, x1:x2]
                    patch_mask = labels[y1:y2, x1:x2]
                    
                    target_pixels = patch_img[patch_mask == label_idx]
                    bg_pixels = patch_img[patch_mask != label_idx]
                    
                    if len(target_pixels) > 0 and len(bg_pixels) > 0:
                        contrast = abs(np.mean(target_pixels) - np.mean(bg_pixels))
                        all_contrasts.append(contrast)
                        
            except Exception as e:
                # 容错处理：防止个别坏样本导致崩溃
                if self.rank == 0:
                    logger.warning(f"Skipping sample {i} during stats calc due to error: {e}")
                continue

        # 计算均值
        s_mean = float(np.mean(all_sizes)) if len(all_sizes) > 0 else 50.0
        c_mean = float(np.mean(all_contrasts)) if len(all_contrasts) > 0 else 50.0
        
        return s_mean, c_mean

    def train_one_epoch_csa(self) -> float:
        """
        Train the model for one complete epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        tag = self.activate_shape_loss
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)

        # >>> 检查是否需要给出形状损失激活提示
        if self.rank == 0 and self.epoch == self.warm_epoch and not self.warned_shape_loss and self.activate_shape_loss:
            logger.warning(
                f"🎉 MiOU >= {self.shape_loss_miou_threshold:.1f}% reached! "
                f"Shape Loss is now activated from Epoch {self.epoch} onwards."
            )
            self.warned_shape_loss = True
            tag = True #
            
        total_loss = 0
        total_samples = 0
        
        if self.rank == 0:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=len(self.train_loader),
                desc=f"[Epoch {self.epoch: 3d}] Training",
                unit="batch",
                color="green",
                leave=False,
            )
        for idx, batch in enumerate(self.train_loader):
            batch_data, batch_masks, _ = batch
            batch_data = batch_data.to(self.device)
            batch_masks = batch_masks.to(self.device)
            pred_logits = self.model(batch_data, tag)
            loss = 0
            for pred_logit in pred_logits:
                loss += self.loss_fn(pred_logit.sigmoid(), batch_masks, input_images=batch_data, activate_shape_loss=self.activate_shape_loss)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)
            if self.rank == 0:
                pbar.update()
                pbar.desc = f"[Training] Epoch {self.epoch:3d} loss={total_loss / total_samples: .6f}"
        if self.rank == 0:
            pbar.close()
        avg_loss = total_loss / total_samples
        if self.scheduler is not None:
            self.scheduler.step()
        # if self.rank == 0:
        #     logger.info(f"[Train][Epoch {self.epoch}] avg_loss={avg_loss:.6f}")
        return avg_loss


    def train_one_epoch_bce(self) -> float:
        """
        Train the model for one complete epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        tag = self.activate_shape_loss
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        total_loss = 0
        total_samples = 0
        if self.rank == 0:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=len(self.train_loader),
                desc=f"[Epoch {self.epoch: 3d}] Training",
                unit="batch",
                color="green",
                leave=False,
            )
        for idx, batch in enumerate(self.train_loader):
            batch_data, batch_masks, _ = batch
            batch_data = batch_data.to(self.device)
            batch_masks = batch_masks.to(self.device)
            pred_logits = self.model(batch_data, tag)
            loss = 0
            for pred_logit in pred_logits:
                loss += self.loss_fn(pred_logit.sigmoid(), batch_masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)
            if self.rank == 0:
                pbar.update()
                pbar.desc = f"[Training] Epoch {self.epoch:3d} loss={total_loss / total_samples: .6f}"
        if self.rank == 0:
            pbar.close()
        avg_loss = total_loss / total_samples
        if self.scheduler is not None:
            self.scheduler.step()
        # if self.rank == 0:
        #     logger.info(f"[Train][Epoch {self.epoch}] avg_loss={avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> Optional[Tuple[float, Optional[metricWrapper]]]:
        """
        Evaluate the model on the validation dataset.
            
        Returns:
            tuple: (average_loss, metric_wrapper) or None if no validation data
        """
        tag = self.activate_shape_loss
        if self.metric_wrapper is not None:
            self.metric_wrapper.reset()
        if self.rank == 0:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=len(self.val_loader),
                desc=f"[Epoch {self.epoch: 3d}] Evaluating",
                unit="batch",
                color="red",
                leave=False,
            )
        self.model.eval()
        if self.val_loader is None:
            return None
        total_loss = 0
        total_samples = 0
        for idx, batch in enumerate(self.val_loader):

            batch_data, batch_masks, _ = batch

            batch_data = batch_data.to(self.device)

            batch_masks = batch_masks.to(self.device)

            loss = 0
            # --- 修改3: 根据 Loss 类型调用不同的计算方式 ---
            if self.loss_type == 'csaloss':
                pred_logit = self.model(batch_data, tag)[0]
             
                loss += self.loss_fn(pred_logit.sigmoid(), batch_masks, input_images=batch_data, activate_shape_loss=tag)
            else:
                pred_logit = self.model(batch_data, tag)[0]
      
                loss += self.loss_fn(pred_logit.sigmoid(), batch_masks)

            total_loss += loss.item() * batch_data.size(0)

            total_samples += batch_data.size(0)
            if self.rank == 0:
                pbar.update()
                pbar.desc = f"[Evaluating] Epoch {self.epoch:3d} loss={total_loss / total_samples: .6f}"
            if self.metric_wrapper:
                self.metric_wrapper(pred_logit, batch_masks)
        avg_loss = total_loss / total_samples

  
        if self.metric_wrapper and self.rank == 0 and self.loss_type == 'csaloss':
            current_miou = get_mIoU(self.metric_wrapper)
            
            if self.epoch >= self.warm_epoch and current_miou >= self.shape_loss_miou_threshold:
                self.activate_shape_loss = True
            
            shape_status = "ON" if self.activate_shape_loss else "OFF"
            logger.info(
                f"[Eval][Epoch {self.epoch}] avg_loss={avg_loss:.6f} metrics: {self.metric_wrapper}"
                f" | Current mIoU: **{current_miou:.2f}** (Threshold: {self.shape_loss_miou_threshold:.1f}, Shape Loss: {shape_status})"
            )
        elif self.metric_wrapper and self.rank == 0:
            # BCE 模式只打印普通日志
            logger.info(f"[Eval][Epoch {self.epoch}] avg_loss={avg_loss:.6f} metrics: {self.metric_wrapper}")
            
        return avg_loss, self.metric_wrapper

    def save_checkpoint(self, save_path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Save training checkpoint to disk.
        
        Only rank 0 process saves checkpoints in distributed training.
        
        Args:
            save_path: Path to save the checkpoint
            extra: Additional data to include in checkpoint (optional)
        """
        if self.rank != 0:
            return
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        state = {
            "epoch": self.epoch,
            "model_state_dict": self.model.module.state_dict()
            if self.distributed
            else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_metric": self.best_metric,
        }
        if extra:
            state.update(extra)
        torch.save(state, save_path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load training checkpoint from disk.
        
        Args:
            path: Path to the checkpoint file
            
        Returns:
            dict: Loaded checkpoint dictionary
        """
        map_location = (
            {"cuda:%d" % 0: "cuda:%d" % self.rank} if self.distributed else self.device
        )
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", None)
        return checkpoint
