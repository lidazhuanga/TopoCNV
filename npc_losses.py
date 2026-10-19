from typing import Dict, List, Sequence, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        weight_bce: float = 0.5,
        weight_dice: float = 0.5,
        include_background: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.include_background = include_background
        self.smooth = smooth

    def _loss_single(
        self,
        logits: torch.Tensor,
        targets_one_hot: torch.Tensor,
        voxel_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Deep Supervision: 如果分辨率不同，下采样标签(nearest)匹配logits分辨率
        # 避免trilinear插值logits产生伪影
        if logits.shape[2:] != targets_one_hot.shape[2:]:
            targets_one_hot = F.interpolate(
                targets_one_hot,
                size=logits.shape[2:],
                mode="nearest",
            )

        # Cross-entropy using softmax (consistent with Dice branch)
        # 使用 softmax 归一化计算 CE，与 Dice 分支保持一致，避免 sigmoid/softmax 冲突
        log_probs = torch.log_softmax(logits, dim=1)
        bce = -(log_probs * targets_one_hot).sum(dim=1).mean()

        probs = torch.softmax(logits, dim=1)
        targets = targets_one_hot
        if not self.include_background and probs.shape[1] > 1:
            probs = probs[:, 1:]
            targets = targets[:, 1:]

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets, dims)
        denom = torch.sum(probs, dims) + torch.sum(targets, dims) + self.smooth
        dice_per_class = 2.0 * intersection / denom
        dice_loss = 1.0 - dice_per_class.mean()

        return self.weight_bce * bce + self.weight_dice * dice_loss

    def forward(self, logits, targets: torch.Tensor) -> torch.Tensor:
        # 支持 deep supervision：logits 可以是 Tensor 或 list/tuple[Tensor]
        if isinstance(logits, (list, tuple)):
            preds: Sequence[torch.Tensor] = list(logits)
        else:
            preds = [logits]

        num_classes = preds[0].shape[1]

        # 处理不同的target格式
        if targets.ndim == 4:
            # [B, D, H, W] -> one-hot [B, C, D, H, W]
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
            targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        elif targets.ndim == 5:
            # 检查是否为 [B, 1, D, H, W] 格式的类别索引
            if targets.shape[1] == 1:
                # 移除通道维度: [B, 1, D, H, W] -> [B, D, H, W]
                targets = targets.squeeze(1)
                # 转换为one-hot
                targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
                targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
            else:
                # 已经是one-hot格式: [B, C, D, H, W]
                targets_one_hot = targets.float()
        else:
            raise ValueError(f"Unexpected target shape {targets.shape}")

        # deep supervision 权重：主输出 1.0，其余 0.5（再归一化）
        weights = [1.0] + [0.5] * (len(preds) - 1)
        weight_sum = float(sum(weights))

        loss = 0.0
        for w, pred in zip(weights, preds):
            loss = loss + w * self._loss_single(pred, targets_one_hot)

        return loss / weight_sum


class WeightedBCEDiceLoss(nn.Module):
    """带类别权重的BCE+Dice损失,针对神经分割优化
    
    针对不同类别的重要性设置权重:
    - 类别1 (CN V - CT可见): 权重最高
    - 类别2 (CN V 0.5 - MRI可见): 权重次之
    - 类别3 (CN V 0.1 - 起始点可见): 权重再次
    - 类别4 (through - 穿行): 权重较低
    - 类别5 (bypass - 绕行): 权重较低
    """
    def __init__(
        self,
        weight_bce: float = 0.5,
        weight_dice: float = 0.5,
        include_background: bool = False,
        smooth: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_focal: bool = True,
    ) -> None:
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.include_background = include_background
        self.smooth = smooth
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 类别权重: [背景, 类别1, 类别2, 类别3, 类别4, 类别5]
        # 根据重要性: 1最重要, 2次之, 3再次, 4和5最不重要
        # 针对极度不平衡数据(前景仅0.027%),大幅提升前景权重
        if class_weights is None:
            # 激进权重: 背景0.1, CN V=10.0, CN V 0.5=8.0, CN V 0.1=6.0, through=3.0, bypass=3.0
            self.class_weights = torch.tensor([0.1, 10.0, 8.0, 6.0, 3.0, 3.0])
        else:
            self.class_weights = class_weights

    def _loss_single(
        self,
        logits: torch.Tensor,
        targets_one_hot: torch.Tensor,
        voxel_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Channel-safe Dice+CE loss (v2 — handles deep supervision with varying output sizes)
        min_ch = getattr(self, '_min_ch_fallback', min(logits.shape[1], targets_one_hot.shape[1]))
        if logits.shape[2:] != targets_one_hot.shape[2:]:
            targets_one_hot = F.interpolate(
                targets_one_hot,
                size=logits.shape[2:],
                mode="nearest",
            )

        # DEBUG: verify file version
        import os; _fver = os.path.getmtime(__file__) if '__file__' in dir() else 0
        # 验证targets_one_hot的有效性
        if torch.isnan(targets_one_hot).any() or torch.isinf(targets_one_hot).any():
            raise ValueError(f"targets_one_hot contains NaN or Inf values")
        
        # Adapt class_weights to match actual number of channels (robust to model variations)
        cw = self.class_weights
        if targets_one_hot.shape[1] != len(cw):
            if targets_one_hot.shape[1] < len(cw):
                cw = cw[:targets_one_hot.shape[1]]
            else:
                pad = torch.ones(targets_one_hot.shape[1] - len(cw))
                cw = torch.cat([cw, pad])
        else:
            cw = self.class_weights

        # 将class_weights移动到正确的设备
        class_weights = cw.to(logits.device)
        
        # 处理体素权重（仅用于BCE/Focal分支，不作用于Dice）
        voxel_weight_expanded = None
        if voxel_weight is not None:
            if voxel_weight.ndim == 4:
                voxel_weight = voxel_weight.unsqueeze(1)  # [B,1,D,H,W]
            elif voxel_weight.ndim != 5:
                raise ValueError(f"Unexpected voxel_weight shape {voxel_weight.shape}")
            if voxel_weight.shape[1] != 1:
                raise ValueError(f"voxel_weight channel must be 1, got {voxel_weight.shape[1]}")
            voxel_weight_expanded = voxel_weight.to(logits.device).expand(-1, logits.shape[1], -1, -1, -1)

        # Focal Loss (替代普通BCE)
        if self.use_focal:
            bce = self._focal_loss(logits, targets_one_hot, class_weights, voxel_weight_expanded)
        else:
            # Cross-entropy using softmax (consistent with Dice branch)
            # 使用 log_softmax 计算 CE，与 Dice 分支保持一致
            log_probs = torch.log_softmax(logits, dim=1)
            # Align channels
            tgt = targets_one_hot
            lp = log_probs
            if lp.shape[1] != tgt.shape[1]:
                min_ch = min(lp.shape[1], tgt.shape[1])
                lp = lp[:, :min_ch]
                tgt = tgt[:, :min_ch]
                cw_exp = class_weights[:min_ch]
            else:
                cw_exp = class_weights
            ce_per_voxel = -(lp * tgt).sum(dim=1)  # [B, D, H, W]
            # 应用类别权重
            class_weights_expanded = cw_exp.view(1, -1, 1, 1, 1)
            weighted_ce = ce_per_voxel * (tgt * class_weights_expanded).sum(dim=1)
            if voxel_weight_expanded is not None:
                voxel_w_3d = voxel_weight_expanded[:, 0, :, :, :] if voxel_weight_expanded.shape[1] > 1 else voxel_weight_expanded.squeeze(1)
                weighted_ce = weighted_ce * voxel_w_3d
            bce = weighted_ce.mean()

        # Weighted Dice Loss
        probs = torch.softmax(logits, dim=1)
        targets = targets_one_hot
        
        # Align channel counts (for deep supervision with varying output sizes)
        min_ch = min(probs.shape[1], targets.shape[1])
        if probs.shape[1] != targets.shape[1]:
            probs = probs[:, :min_ch]
            targets = targets[:, :min_ch]
        
        if not self.include_background and probs.shape[1] > 1:
            probs = probs[:, 1:]
            targets = targets[:, 1:]
            # Use n_fg directly — avoids any stale-bytecode min_ch issues
            n_fg = probs.shape[1]
            if 'min_ch' in dir() and min_ch > 1:
                w_start = 1
            else:
                w_start = 1
            if n_fg <= len(class_weights) - w_start:
                weights_for_dice = class_weights[w_start:w_start+n_fg]
            else:
                weights_for_dice = class_weights[w_start:]
        else:
            weights_for_dice = class_weights[:probs.shape[1]]

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets, dims)
        denom = torch.sum(probs, dims) + torch.sum(targets, dims) + self.smooth
        dice_per_class = 2.0 * intersection / denom
        
        # 应用类别权重到Dice
        weighted_dice = (dice_per_class * weights_for_dice).sum() / weights_for_dice.sum()
        dice_loss = 1.0 - weighted_dice

        return self.weight_bce * bce + self.weight_dice * dice_loss
    
    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor,
        voxel_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Multi-class Focal Loss using softmax (consistent with Dice branch).
        
        使用 softmax 而非 sigmoid，与 Dice 分支的归一化方式保持一致，
        避免两个 loss 分支产生冲突的梯度方向。
        对于互斥的多类分割，softmax 确保各类别概率之和为 1，
        梯度信号与 Dice+softmax 分支协同。
        """
        # Softmax 概率与 log 概率（与 Dice 分支一致）
        probs = torch.softmax(logits, dim=1)      # [B, C, D, H, W]
        log_probs = torch.log_softmax(logits, dim=1)  # [B, C, D, H, W] 数值稳定
        
        # Align channel counts: targets may differ from logits due to model variations
        if probs.shape[1] != targets.shape[1]:
            tgt_ch = targets.shape[1]
            if probs.shape[1] > tgt_ch:
                probs = probs[:, :tgt_ch]
                logits = logits[:, :tgt_ch]
                log_probs = log_probs[:, :tgt_ch]
            else:
                targets = targets[:, :probs.shape[1]]
        
        # p_t: 每个体素属于其真实类别的概率 [B, D, H, W]
        p_t = (probs * targets).sum(dim=1).clamp(min=1e-7, max=1.0 - 1e-7)
        
        # Focal weight: (1 - p_t)^gamma — 容易样本(p_t高)权重低，困难样本(p_t低)权重高
        focal_weight = (1.0 - p_t) ** self.focal_gamma  # [B, D, H, W]
        
        # Cross-entropy per voxel: -log(p_t), 使用 log_softmax 保证数值稳定性
        ce = -(log_probs * targets).sum(dim=1)  # [B, D, H, W]
        
        # 每个体素的类别权重（根据其真实类别）
        class_weight_per_voxel = (targets * class_weights.view(1, -1, 1, 1, 1)).sum(dim=1)  # [B, D, H, W]
        
        # Focal loss with alpha balancing
        focal_loss = self.focal_alpha * focal_weight * ce * class_weight_per_voxel
        
        # 体素级权重（如不确定性图），形状适配为 [B, D, H, W]
        if voxel_weight is not None:
            if voxel_weight.shape[1] == 1:
                voxel_weight = voxel_weight.squeeze(1)
            elif voxel_weight.shape[1] > 1:
                voxel_weight = voxel_weight[:, 0, :, :, :]
            focal_loss = focal_loss * voxel_weight
        
        return focal_loss.mean()

    def forward(self, logits, targets: torch.Tensor, voxel_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 支持 deep supervision：logits 可以是 Tensor 或 list/tuple[Tensor]
        if isinstance(logits, (list, tuple)):
            preds: Sequence[torch.Tensor] = list(logits)
        else:
            preds = [logits]

        num_classes = preds[0].shape[1]

        # 处理不同的target格式
        if targets.ndim == 4:
            # [B, D, H, W] -> one-hot [B, C, D, H, W]
            # 验证标签值范围
            min_val = targets.min().item()
            max_val = targets.max().item()
            if min_val < 0 or max_val >= num_classes:
                raise ValueError(
                    f"Label values out of range: min={min_val}, max={max_val}, "
                    f"but num_classes={num_classes} (valid range: 0-{num_classes-1}). "
                    f"Target shape: {targets.shape}"
                )
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
            targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        elif targets.ndim == 5:
            # 检查是否为 [B, 1, D, H, W] 格式的类别索引
            if targets.shape[1] == 1:
                # 移除通道维度: [B, 1, D, H, W] -> [B, D, H, W]
                targets = targets.squeeze(1)
                # 验证标签值范围
                min_val = targets.min().item()
                max_val = targets.max().item()
                if min_val < 0 or max_val >= num_classes:
                    raise ValueError(
                        f"Label values out of range: min={min_val}, max={max_val}, "
                        f"but num_classes={num_classes} (valid range: 0-{num_classes-1}). "
                        f"Target shape: {targets.shape}"
                    )
                # 转换为one-hot
                targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
                targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
            else:
                # 已经是one-hot格式: [B, C, D, H, W]
                targets_one_hot = targets.float()
        else:
            raise ValueError(f"Unexpected target shape {targets.shape}")

        # deep supervision 权重：主输出 1.0，其余 0.5（再归一化）
        weights = [1.0] + [0.5] * (len(preds) - 1)
        weight_sum = float(sum(weights))

        loss = 0.0
        for w, pred in zip(weights, preds):
            loss = loss + w * self._loss_single(pred, targets_one_hot, voxel_weight=voxel_weight)

        return loss / weight_sum


def multiclass_dice_score(
    logits,
    targets: torch.Tensor,
    include_background: bool = False,
    smooth: float = 1e-5,
    per_class: bool = False,
) -> float:
    # 只用主输出计算 Dice
    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    num_classes = logits.shape[1]

    # 处理不同的target格式
    if targets.ndim == 4:
        # [B, D, H, W] -> one-hot [B, C, D, H, W]
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
    elif targets.ndim == 5:
        # 检查是否为 [B, 1, D, H, W] 格式的类别索引
        if targets.shape[1] == 1:
            # 移除通道维度: [B, 1, D, H, W] -> [B, D, H, W]
            targets = targets.squeeze(1)
            # 转换为one-hot
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
            targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        else:
            # 已经是one-hot格式: [B, C, D, H, W]
            targets_one_hot = targets.float()
    else:
        raise ValueError(f"Unexpected target shape {targets.shape}")

    probs = torch.softmax(logits, dim=1)
    # Align channel counts for robustness (model output may vary in deep supervision)
    if probs.shape[1] != targets_one_hot.shape[1]:
        min_ch = min(probs.shape[1], targets_one_hot.shape[1])
        probs = probs[:, :min_ch]
        targets_one_hot = targets_one_hot[:, :min_ch]
    if not include_background and num_classes > 1:
        probs = probs[:, 1:]
        targets_one_hot = targets_one_hot[:, 1:]

    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * targets_one_hot, dims)
    denom = torch.sum(probs, dims) + torch.sum(targets_one_hot, dims) + smooth
    dice_per_class = 2.0 * intersection / denom
    
    if per_class:
        return dice_per_class.cpu().numpy()
    
    dice_mean = dice_per_class.mean().item()
    return float(dice_mean)


# ==================== Hierarchical Tree Loss ====================


class HierarchicalTreeLoss(nn.Module):
    """Multi-level loss for the Hierarchical Soft-Tree Stage2 model.

    L_total = L_final
            + lambda_group  * L_group
            + lambda_branch * L_branch_A
            + lambda_aux    * L_aux

    - L_final:   CE + Dice on the 5-class tree-fused output (main supervision).
                 Classes: {0:bg, 1:c1, 2:c2, 3:c3, 4:c5}
    - L_group:   CE + Dice on the 3-class group head vs coarse labels
                 {bg=0, nerve_body=1, class5=2}.
    - L_branch_A:  Soft-masked CE + Dice on conditional branch output
                 (class 1/2/3 sub-classification).
    - L_aux:     Binary BCE + Dice for class4 (auxiliary overlapping channel).
                 Class4 is the transit zone that overlaps with class1.
    """

    def __init__(
        self,
        final_loss: nn.Module,
        group_loss: nn.Module,
        branch_loss: nn.Module,
        lambda_group: float = 0.3,
        lambda_branch: float = 0.2,
        lambda_aux: float = 0.3,
    ) -> None:
        super().__init__()
        self.final_loss = final_loss        # WeightedBCEDiceLoss(5-class)
        self.group_loss = group_loss        # WeightedBCEDiceLoss(3-class)
        self.branch_loss = branch_loss      # WeightedBCEDiceLoss for branch A
        self.lambda_group = lambda_group
        self.lambda_branch = lambda_branch
        self.lambda_aux = lambda_aux

    @staticmethod
    def _labels_to_group(labels: torch.Tensor) -> torch.Tensor:
        """Convert main labels [B,D,H,W] in {0..4} to group labels {0,1,2}.

        Main labels are already remapped: {0:bg, 1:c1, 2:c2, 3:c3, 4:c5}
        Group mapping:
          0 -> 0 (bg)
          1,2,3 -> 1 (nerve_body)
          4 -> 2 (class5)
        """
        group = torch.zeros_like(labels)
        group[(labels >= 1) & (labels <= 3)] = 1
        group[labels == 4] = 2
        return group.long()

    @staticmethod
    def _labels_to_branch_a(labels: torch.Tensor) -> torch.Tensor:
        """Extract nerve_body sub-class labels from main labels.

        Main labels 1/2/3 -> sub 0/1/2.
        All other voxels -> 0 (will be masked out by soft mask).
        """
        sub = torch.zeros_like(labels)
        mask = (labels >= 1) & (labels <= 3)
        sub[mask] = labels[mask] - 1   # 1->0, 2->1, 3->2
        return sub.long()

    @staticmethod
    def _binary_aux_loss(aux_logits: torch.Tensor, aux_target: torch.Tensor) -> torch.Tensor:
        """Binary BCE + Dice loss for auxiliary class4 head.

        Args:
            aux_logits: [B, 1, D, H, W] raw logits (pre-sigmoid)
            aux_target: [B, D, H, W] binary target (0 or 1)
        """
        pred = torch.sigmoid(aux_logits.squeeze(1))  # [B, D, H, W]
        target = aux_target.float()

        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(
            aux_logits.squeeze(1), target, reduction='mean'
        )

        # Dice
        smooth = 1e-5
        intersection = (pred * target).sum()
        denom = pred.sum() + target.sum() + smooth
        dice_loss = 1.0 - 2.0 * intersection / denom

        return 0.4 * bce + 0.6 * dice_loss

    def _soft_masked_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        soft_mask: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CE + Dice with per-voxel soft weighting.

        Args:
            logits:        [B, C, D, H, W]
            targets:       [B, D, H, W]  (class indices)
            soft_mask:     [B, D, H, W]  (weights in [0, 1])
            class_weights: Optional [C] tensor of per-class weights for both
                           CE (per-voxel multiplication) and Dice (weighted mean).
        """
        num_classes = logits.shape[1]
        target_oh = F.one_hot(targets.long(), num_classes).permute(0, 4, 1, 2, 3).float()
        log_probs = torch.log_softmax(logits, dim=1)  # [B, C, D, H, W]
        ce_per_voxel = -(log_probs * target_oh).sum(dim=1)  # [B, D, H, W]

        if class_weights is not None:
            cw = class_weights[:num_classes].to(logits.device)
            class_weight_per_voxel = (target_oh * cw.view(1, -1, 1, 1, 1)).sum(dim=1)
            ce_per_voxel = ce_per_voxel * class_weight_per_voxel

        ce = (ce_per_voxel * soft_mask).sum() / (soft_mask.sum() + 1e-8)

        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3, 4)
        intersection = (probs * target_oh * soft_mask.unsqueeze(1)).sum(dims)
        denom = (probs * soft_mask.unsqueeze(1)).sum(dims) + (target_oh * soft_mask.unsqueeze(1)).sum(dims) + 1e-5
        dice_per_class = 2.0 * intersection / denom

        if class_weights is not None:
            weighted_dice = (dice_per_class * cw).sum() / cw.sum()
            dice_loss = 1.0 - weighted_dice
        else:
            dice_loss = 1.0 - dice_per_class.mean()

        return 0.4 * ce + 0.6 * dice_loss

    def forward(
        self,
        final_probs: torch.Tensor,       # [B, 5, D, H, W] – already a probability map
        group_logits: torch.Tensor,      # [B, 3, D, H, W]
        branch_a_logits: torch.Tensor,   # [B, 3, D, H, W]
        aux_logits: torch.Tensor,        # [B, 1, D, H, W] – class4 auxiliary logits
        labels: torch.Tensor,            # [B, D, H, W] main labels in {0..4}
        aux_target: torch.Tensor,        # [B, D, H, W] binary class4 target
        p_group: torch.Tensor = None,    # [B, 3, D, H, W] – blended group probs for soft mask
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components.

        If *p_group* is provided it is used as the soft mask for branch losses.
        Otherwise the softmax of *group_logits* is used.
        """
        # --- L_final: convert final_probs to pseudo-logits for the existing loss ---
        # WeightedBCEDiceLoss applies softmax internally.  Since final_probs is
        # already a probability distribution (sums to 1), we use log(p) as
        # pseudo-logits: softmax(log(p)) == p  when  sum(p) == 1.
        final_logits = torch.log(final_probs.clamp(1e-7, 1.0))
        l_final = self.final_loss(final_logits, labels)

        # --- L_group ---
        group_labels = self._labels_to_group(labels)
        l_group = self.group_loss(group_logits, group_labels)

        # --- Soft mask for branch A ---
        if p_group is not None:
            mask_a = p_group[:, 1]  # P(nerve_body)
        else:
            pg = torch.softmax(group_logits, dim=1)
            mask_a = pg[:, 1]

        # --- L_branch_A --- (class 1/2/3 → sub 0/1/2, decreasing importance)
        branch_a_labels = self._labels_to_branch_a(labels)
        branch_a_weights = torch.tensor([3.0, 2.0, 1.0], device=labels.device)
        l_branch_a = self._soft_masked_loss(branch_a_logits, branch_a_labels, mask_a,
                                            class_weights=branch_a_weights)

        # --- L_aux --- (class4 auxiliary binary loss)
        l_aux = self._binary_aux_loss(aux_logits, aux_target)

        total = (l_final
                 + self.lambda_group * l_group
                 + self.lambda_branch * l_branch_a
                 + self.lambda_aux * l_aux)

        return {
            'total': total,
            'l_final': l_final,
            'l_group': l_group,
            'l_branch_a': l_branch_a,
            'l_aux': l_aux,
        }


# ==================== 可微拓扑损失 ====================


def _soft_skel3d(
    probs: torch.Tensor,
    num_iter: int = 10,
) -> torch.Tensor:
    """
    可微 3D 软骨架化 (soft skeletonization)。

    基于迭代 min-pooling + max-pooling 实现，参考:
      Shit et al., "clDice - a Novel Topology-Preserving Loss Function
      for Tubular Structure Segmentation", CVPR 2021.

    Args:
        probs:   [B, 1, D, H, W] 前景概率图 (或 [B, D, H, W])
        num_iter: 迭代次数

    Returns:
        soft_skel: 与 probs 同 shape
    """
    if probs.ndim == 4:
        probs = probs.unsqueeze(1)  # [B, 1, D, H, W]
    B, C, D, H, W = probs.shape

    # 3D max-pooling (3x3x3, stride=1, padding=1)
    mp = torch.nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
    # 3D min-pooling = -max_pool(-x)
    def min_pool3d(x):
        return -mp(-x)

    skel = probs
    for _ in range(num_iter):
        # 形态学腐蚀: min-pool then max-pool
        eroded = min_pool3d(skel)
        # 形态学膨胀: max-pool
        dilated = mp(eroded)
        # 保留被腐蚀掉的边界 = 骨架增量
        boundary = torch.relu(skel - dilated)
        skel = boundary

    if C == 1:
        skel = skel.squeeze(1)  # [B, D, H, W]
    return skel


class SoftClDiceLoss(nn.Module):
    """
    可微 clDice 损失。

    在预测概率图和 GT 二值图上分别做软骨架化，
    然后计算交叉配对 Dice。

    L_cldice = 1 - clDice
    clDice = 2 * Tprec * Tsens / (Tprec + Tsens + eps)
      Tprec = |V_L ∩ S_P| / |S_P|   (GT 骨架在预测体积中的占比)
      Tsens = |V_P ∩ S_L| / |S_L|   (预测骨架在 GT 体积中的占比)
    """

    def __init__(
        self,
        num_iter: int = 10,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth

    def forward(
        self,
        pred_probs: torch.Tensor,   # [B, C, D, H, W] 或 [B, D, H, W]
        gt_binary: torch.Tensor,     # [B, D, H, W] bool/int 或 [B, 1, D, H, W]
    ) -> torch.Tensor:
        """
        计算 clDice 损失。

        Args:
            pred_probs: 预测概率图 (多类时取前景通道)
            gt_binary:  GT 二值 mask
        """
        # 提取前景概率
        if pred_probs.ndim == 5 and pred_probs.shape[1] > 1:
            # 多类: 取 class 1-4 的总概率
            pred_fg = pred_probs[:, 1:].sum(dim=1)  # [B, D, H, W]
        elif pred_probs.ndim == 5:
            pred_fg = pred_probs[:, 0]  # [B, D, H, W]
        else:
            pred_fg = pred_probs

        if gt_binary.ndim == 5:
            gt_binary = gt_binary.squeeze(1)
        gt_fg = gt_binary.float()

        # 软骨架化
        pred_skel = _soft_skel3d(pred_fg, self.num_iter)  # [B, D, H, W]
        gt_skel = _soft_skel3d(gt_fg, self.num_iter)      # [B, D, H, W]

        # Tprec: GT 骨架与预测体积的重叠
        tprec = (gt_skel * pred_fg).sum(dim=(1, 2, 3)) / (gt_skel.sum(dim=(1, 2, 3)) + self.smooth)
        # Tsens: 预测骨架与 GT 体积的重叠
        tsens = (pred_skel * gt_fg).sum(dim=(1, 2, 3)) / (pred_skel.sum(dim=(1, 2, 3)) + self.smooth)

        cldice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        return (1.0 - cldice).mean()


class TemplateGraphTopologyLoss(nn.Module):
    """
    解剖模板图拓扑损失。

    三个组件:
      L_keypoint:   预测热力图 → soft-argmax → 与 GT 关键点坐标的 MSE
      L_graph_edge: 模板边连接的两关键点之间，沿线段采样分割概率 → 期望全为前景
      L_branch_dir: 分支体素的空间梯度与模板方向场的余弦相似度

    Patch 训练时使用部分图监督:
      只对当前 patch 内 GT 存在的关键点和边计算损失。
    """

    def __init__(
        self,
        lambda_keypoint: float = 1.0,
        lambda_graph: float = 1.0,
        lambda_branch_dir: float = 0.5,
        num_keypoints: int = 5,
        template_edges: List[Tuple[int, int]] = None,
        edge_sample_points: int = 32,
        keypoint_sigma: float = 4.0,
    ) -> None:
        """
        Args:
            lambda_keypoint:   关键点定位损失权重
            lambda_graph:      图边完整性损失权重
            lambda_branch_dir: 分支方向损失权重
            num_keypoints:     关键点数量 K
            template_edges:   模板边列表 [(i,j), ...]
            edge_sample_points: 每条边采样点数
            keypoint_sigma:    soft-argmax 温度参数 (越小越尖锐)
        """
        super().__init__()
        self.lambda_keypoint = lambda_keypoint
        self.lambda_graph = lambda_graph
        self.lambda_branch_dir = lambda_branch_dir
        self.num_keypoints = num_keypoints
        self.template_edges = template_edges or [(0, 1), (1, 2), (1, 3), (1, 4)]
        self.edge_sample_points = edge_sample_points
        self.keypoint_sigma = keypoint_sigma

    @staticmethod
    def _soft_argmax_3d(
        heatmap: torch.Tensor,
        beta: float = 100.0,
    ) -> torch.Tensor:
        """
        从热力图 [K, D, H, W] 用 soft-argmax 提取坐标 [K, 3]。

        soft-argmax = sum(p_i * coord_i), 其中 p_i = softmax(beta * heatmap_i)
        """
        K, D, H, W = heatmap.shape
        device = heatmap.device

        # 创建坐标网格
        zz = torch.arange(D, device=device, dtype=torch.float32)
        yy = torch.arange(H, device=device, dtype=torch.float32)
        xx = torch.arange(W, device=device, dtype=torch.float32)
        zg, yg, xg = torch.meshgrid(zz, yy, xx, indexing='ij')  # (D, H, W)

        # Softmax 归一化
        h_flat = heatmap.view(K, -1)  # [K, D*H*W]
        h_soft = torch.softmax(h_flat * beta, dim=1)  # [K, D*H*W]
        h_soft = h_soft.view(K, D, H, W)

        # 加权求和提取坐标
        z_coord = (h_soft * zg.unsqueeze(0)).sum(dim=(1, 2, 3))  # [K]
        y_coord = (h_soft * yg.unsqueeze(0)).sum(dim=(1, 2, 3))
        x_coord = (h_soft * xg.unsqueeze(0)).sum(dim=(1, 2, 3))

        return torch.stack([z_coord, y_coord, x_coord], dim=1)  # [K, 3]

    @staticmethod
    def _sample_line_3d(
        volume: torch.Tensor,  # [C, D, H, W] 或 [D, H, W]
        p_start: torch.Tensor,  # [3]
        p_end: torch.Tensor,    # [3]
        num_samples: int = 32,
    ) -> torch.Tensor:
        """
        沿 p_start → p_end 线段在 volume 中做三线性插值采样。

        Returns:
            samples: [num_samples, C] 或 [num_samples]
        """
        t = torch.linspace(0.0, 1.0, num_samples, device=volume.device)  # [N]
        # 归一化到 [-1, 1]
        D, H, W = volume.shape[-3:]
        points_z = (p_start[0] + t * (p_end[0] - p_start[0])) / (D - 1) * 2.0 - 1.0
        points_y = (p_start[1] + t * (p_end[1] - p_start[1])) / (H - 1) * 2.0 - 1.0
        points_x = (p_start[2] + t * (p_end[2] - p_start[2])) / (W - 1) * 2.0 - 1.0
        grid = torch.stack([points_x, points_y, points_z], dim=1)  # [N, 3]
        # grid_sample 3D expects grid shape [B, D_out, H_out, W_out, 3]
        # We want 1D sampling: D_out=1, H_out=1, W_out=N
        grid = grid.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # [1, 1, 1, N, 3]

        if volume.ndim == 3:
            vol = volume.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            sampled = F.grid_sample(vol, grid, mode='bilinear', padding_mode='border', align_corners=True)
            # sampled shape: [1, 1, 1, 1, N] → squeeze → [N]
            return sampled.squeeze()
        elif volume.ndim == 4:
            vol = volume.unsqueeze(0)  # [1, C, D, H, W]
            sampled = F.grid_sample(vol, grid, mode='bilinear', padding_mode='border', align_corners=True)
            # sampled shape: [1, C, 1, 1, N] → [C, N] → [N, C]
            return sampled[0, :, 0, 0, :].t()
        else:
            raise ValueError(f"Unexpected volume ndim: {volume.ndim}")

    def forward(
        self,
        seg_logits: torch.Tensor,         # [B, C, D, H, W] 分割 logits
        keypoint_heatmaps: torch.Tensor,   # [B, K, D, H, W] 预测热力图
        gt_keypoint_coords: torch.Tensor,  # [B, K, 3] GT 关键点坐标 (< 0 表示不在 patch 内)
        gt_topo_coords: torch.Tensor,      # [B, D, H, W, 3] GT 拓扑坐标
        gt_branch_dirs: torch.Tensor,      # [B, D, H, W, 3] GT 分支方向场
        gt_fg_mask: torch.Tensor = None,   # [B, D, H, W] 前景 mask (可选)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with:
              - 'total':         标量总损失
              - 'loss_keypoint': 关键点定位损失
              - 'loss_graph':    图边完整性损失
              - 'loss_branch_dir': 分支方向损失
        """
        B, C, D, H, W = seg_logits.shape
        device = seg_logits.device

        # ---- L_keypoint: 关键点定位 -------
        loss_keypoint = torch.tensor(0.0, device=device)
        valid_batches = 0

        for b in range(B):
            # 提取预测关键点 (soft-argmax)
            pred_coords = self._soft_argmax_3d(
                keypoint_heatmaps[b], beta=1.0 / max(self.keypoint_sigma, 1e-3)
            )  # [K, 3]

            gt_coords = gt_keypoint_coords[b]  # [K, 3]

            # 只对 GT 有效 (>= 0) 的关键点计算损失
            valid_mask = (gt_coords[:, 0] >= 0) & (gt_coords[:, 1] >= 0) & (gt_coords[:, 2] >= 0)
            if valid_mask.any():
                diff = pred_coords[valid_mask] - gt_coords[valid_mask]
                # 归一化: 除以体积尺寸使不同分辨率可比
                norm_factor = torch.tensor([D, H, W], device=device, dtype=torch.float32)
                diff_norm = diff / norm_factor.unsqueeze(0)
                loss_keypoint += (diff_norm ** 2).mean()
                valid_batches += 1

        if valid_batches > 0:
            loss_keypoint = loss_keypoint / valid_batches

        # ---- L_graph_edge: 图边完整性 -------
        loss_graph = torch.tensor(0.0, device=device)
        valid_edges_total = 0
        seg_probs = torch.softmax(seg_logits, dim=1)  # [B, C, D, H, W]

        for b in range(B):
            gt_coords = gt_keypoint_coords[b]  # [K, 3]
            pred_coords = self._soft_argmax_3d(
                keypoint_heatmaps[b], beta=1.0 / max(self.keypoint_sigma, 1e-3)
            )

            for edge_i, edge_j in self.template_edges:
                # 检查两个端点是否在 patch 内
                gti = gt_coords[edge_i]
                gtj = gt_coords[edge_j]
                if gti[0] < 0 or gtj[0] < 0:
                    continue  # 至少一个端点不在 patch 内，跳过

                pi = pred_coords[edge_i]
                pj = pred_coords[edge_j]

                # 沿线段采样前景概率 (class 1-4 的总概率)
                fg_prob = seg_probs[b, 1:].sum(dim=0)  # [D, H, W]
                sampled = self._sample_line_3d(fg_prob, pi, pj, self.edge_sample_points)
                # 线段上的概率应该都接近 1 (存在神经)
                loss_graph += F.binary_cross_entropy(
                    sampled.clamp(1e-7, 1 - 1e-7),
                    torch.ones_like(sampled),
                )
                valid_edges_total += 1

        if valid_edges_total > 0:
            loss_graph = loss_graph / valid_edges_total

        # ---- L_branch_dir: 分支方向一致性 -------
        loss_branch_dir = torch.tensor(0.0, device=device)
        valid_voxels = 0

        for b in range(B):
            # 前景体素: topology_coords[:,:,:,0] >= 0
            bid_map = gt_topo_coords[b, ..., 0]  # [D, H, W]
            fg_voxels = (bid_map >= 0)
            if not fg_voxels.any():
                continue

            # 提取前景体素的位置
            fg_indices = torch.nonzero(fg_voxels, as_tuple=False)  # [N, 3]
            if len(fg_indices) == 0:
                continue
            # 限制采样数量以避免 OOM
            max_samples = 8192
            if len(fg_indices) > max_samples:
                perm = torch.randperm(len(fg_indices), device=device)[:max_samples]
                fg_indices = fg_indices[perm]

            N = len(fg_indices)
            valid_voxels += N

            # GT 分支方向
            gt_dir = gt_branch_dirs[b, fg_indices[:, 0], fg_indices[:, 1], fg_indices[:, 2]]  # [N, 3]
            gt_dir_norm = gt_dir / (gt_dir.norm(dim=1, keepdim=True) + 1e-8)

            # 计算 seg_logits 在这些体素处的空间梯度
            fg_prob_b = seg_probs[b, 1:].sum(dim=0)  # [D, H, W]
            fg_prob_b = fg_prob_b.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

            # 用 3D sobel 近似梯度
            grad_z = fg_prob_b[:, :, 2:, :, :] - fg_prob_b[:, :, :-2, :, :]
            grad_y = fg_prob_b[:, :, :, 2:, :] - fg_prob_b[:, :, :, :-2, :]
            grad_x = fg_prob_b[:, :, :, :, 2:] - fg_prob_b[:, :, :, :, :-2]
            # padding 回去
            grad_z = F.pad(grad_z, (0, 0, 0, 0, 1, 1))
            grad_y = F.pad(grad_y, (0, 0, 1, 1, 0, 0))
            grad_x = F.pad(grad_x, (1, 1, 0, 0, 0, 0))

            grad_z = grad_z.squeeze(0).squeeze(0)  # [D, H, W]
            grad_y = grad_y.squeeze(0).squeeze(0)
            grad_x = grad_x.squeeze(0).squeeze(0)

            pred_grad = torch.stack([
                grad_z[fg_indices[:, 0], fg_indices[:, 1], fg_indices[:, 2]],
                grad_y[fg_indices[:, 0], fg_indices[:, 1], fg_indices[:, 2]],
                grad_x[fg_indices[:, 0], fg_indices[:, 1], fg_indices[:, 2]],
            ], dim=1)  # [N, 3]

            pred_grad_norm = pred_grad / (pred_grad.norm(dim=1, keepdim=True) + 1e-8)

            # 余弦相似度损失
            cos_sim = (pred_grad_norm * gt_dir_norm).sum(dim=1)
            loss_branch_dir += (1.0 - cos_sim).mean()

        if valid_voxels > 0:
            loss_branch_dir = loss_branch_dir / B

        # ---- 组合 -------
        total = (
            self.lambda_keypoint * loss_keypoint
            + self.lambda_graph * loss_graph
            + self.lambda_branch_dir * loss_branch_dir
        )

        return {
            'total': total,
            'loss_keypoint': loss_keypoint,
            'loss_graph': loss_graph,
            'loss_branch_dir': loss_branch_dir,
        }


class CombinedTopologyLoss(nn.Module):
    """
    组合损失: WeightedBCEDiceLoss + SoftClDiceLoss + TemplateGraphTopologyLoss。

    用法:
        combined = CombinedTopologyLoss(
            seg_loss=WeightedBCEDiceLoss(...),
            lambda_cldice=1.0,
            lambda_graph=1.0,
            lambda_keypoint=1.0,
            lambda_branch_dir=0.5,
        )
        loss_dict = combined(seg_logits, labels, keypoint_heatmaps, gt_keypoints, gt_topo, gt_dirs)
        loss = loss_dict['total']
    """

    def __init__(
        self,
        seg_loss: nn.Module,
        lambda_cldice: float = 1.0,
        lambda_graph: float = 1.0,
        lambda_keypoint: float = 1.0,
        lambda_branch_dir: float = 0.5,
        num_keypoints: int = 5,
        template_edges: List[Tuple[int, int]] = None,
        cldice_num_iter: int = 10,
    ) -> None:
        super().__init__()
        self.seg_loss = seg_loss
        self.lambda_cldice = lambda_cldice
        self.cldice_loss = SoftClDiceLoss(num_iter=cldice_num_iter)
        self.graph_loss = TemplateGraphTopologyLoss(
            lambda_keypoint=lambda_keypoint,
            lambda_graph=lambda_graph,
            lambda_branch_dir=lambda_branch_dir,
            num_keypoints=num_keypoints,
            template_edges=template_edges,
        )

    def forward(
        self,
        seg_logits,                       # [B, C, D, H, W]
        labels,                            # [B, D, H, W] or [B, C, D, H, W]
        keypoint_heatmaps: Optional[torch.Tensor] = None,   # [B, K, D, H, W]
        gt_keypoint_coords: Optional[torch.Tensor] = None,  # [B, K, 3]
        gt_topo_coords: Optional[torch.Tensor] = None,      # [B, D, H, W, 3]
        gt_branch_dirs: Optional[torch.Tensor] = None,       # [B, D, H, W, 3]
        voxel_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_seg = self.seg_loss(seg_logits, labels, voxel_weight=voxel_weight)
        total = loss_seg
        losses = {'loss_seg': loss_seg, 'loss_cldice': torch.tensor(0.0, device=seg_logits.device if isinstance(seg_logits, torch.Tensor) else loss_seg.device),
                  'loss_keypoint': torch.tensor(0.0),
                  'loss_graph': torch.tensor(0.0),
                  'loss_branch_dir': torch.tensor(0.0)}

        device = loss_seg.device

        # SoftClDiceLoss
        if self.lambda_cldice > 0 and labels is not None:
            seg_probs = torch.softmax(seg_logits if not isinstance(seg_logits, (list, tuple)) else seg_logits[0], dim=1)
            if labels.ndim == 4:
                gt_binary = ((labels >= 1) & (labels <= 4)).float()
            else:
                # multi-hot [B, C, D, H, W]
                gt_binary = labels[:, 1:].sum(dim=1).clamp(0, 1)
            loss_cldice = self.cldice_loss(seg_probs, gt_binary)
            total = total + self.lambda_cldice * loss_cldice
            losses['loss_cldice'] = loss_cldice.to(device)

        # TemplateGraphTopologyLoss
        if keypoint_heatmaps is not None and gt_keypoint_coords is not None:
            graph_losses = self.graph_loss(
                seg_logits if not isinstance(seg_logits, (list, tuple)) else seg_logits[0],
                keypoint_heatmaps,
                gt_keypoint_coords,
                gt_topo_coords if gt_topo_coords is not None else torch.zeros(1),
                gt_branch_dirs if gt_branch_dirs is not None else torch.zeros(1),
            )
            total = total + graph_losses['total']
            losses['loss_keypoint'] = graph_losses['loss_keypoint'].to(device)
            losses['loss_graph'] = graph_losses['loss_graph'].to(device)
            losses['loss_branch_dir'] = graph_losses['loss_branch_dir'].to(device)

        losses['total'] = total
        return losses
