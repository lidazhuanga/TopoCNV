"""Differentiable topology-aware losses used by TopoCNV."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_skel3d(
    probs: torch.Tensor,
    num_iter: int = 10,
) -> torch.Tensor:
    """Compute the legacy differentiable 3D soft skeletonization."""
    if probs.ndim == 4:
        probs = probs.unsqueeze(1)
    _, channels, _, _, _ = probs.shape
    max_pool = torch.nn.MaxPool3d(kernel_size=3, stride=1, padding=1)

    def min_pool3d(tensor):
        return -max_pool(-tensor)

    skeleton = probs
    for _ in range(num_iter):
        eroded = min_pool3d(skeleton)
        dilated = max_pool(eroded)
        boundary = torch.relu(skeleton - dilated)
        skeleton = boundary
    if channels == 1:
        skeleton = skeleton.squeeze(1)
    return skeleton


class SoftClDiceLoss(nn.Module):
    """Differentiable clDice objective for tubular foreground structures."""

    def __init__(self, num_iter: int = 10, smooth: float = 1e-5) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth

    def forward(
        self,
        pred_probs: torch.Tensor,
        gt_binary: torch.Tensor,
    ) -> torch.Tensor:
        if pred_probs.ndim == 5 and pred_probs.shape[1] > 1:
            pred_fg = pred_probs[:, 1:].sum(dim=1)
        elif pred_probs.ndim == 5:
            pred_fg = pred_probs[:, 0]
        else:
            pred_fg = pred_probs
        if gt_binary.ndim == 5:
            gt_binary = gt_binary.squeeze(1)
        gt_fg = gt_binary.float()

        pred_skel = _soft_skel3d(pred_fg, self.num_iter)
        gt_skel = _soft_skel3d(gt_fg, self.num_iter)
        tprec = (gt_skel * pred_fg).sum(dim=(1, 2, 3)) / (
            gt_skel.sum(dim=(1, 2, 3)) + self.smooth
        )
        tsens = (pred_skel * gt_fg).sum(dim=(1, 2, 3)) / (
            pred_skel.sum(dim=(1, 2, 3)) + self.smooth
        )
        cldice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        return (1.0 - cldice).mean()


class TemplateGraphTopologyLoss(nn.Module):
    """Anatomical template graph topology objective."""

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
        keypoints, depth, height, width = heatmap.shape
        device = heatmap.device
        z_axis = torch.arange(depth, device=device, dtype=torch.float32)
        y_axis = torch.arange(height, device=device, dtype=torch.float32)
        x_axis = torch.arange(width, device=device, dtype=torch.float32)
        z_grid, y_grid, x_grid = torch.meshgrid(
            z_axis, y_axis, x_axis, indexing="ij"
        )
        heatmap_soft = torch.softmax(heatmap.view(keypoints, -1) * beta, dim=1)
        heatmap_soft = heatmap_soft.view(keypoints, depth, height, width)
        z_coord = (heatmap_soft * z_grid.unsqueeze(0)).sum(dim=(1, 2, 3))
        y_coord = (heatmap_soft * y_grid.unsqueeze(0)).sum(dim=(1, 2, 3))
        x_coord = (heatmap_soft * x_grid.unsqueeze(0)).sum(dim=(1, 2, 3))
        return torch.stack([z_coord, y_coord, x_coord], dim=1)

    @staticmethod
    def _sample_line_3d(
        volume: torch.Tensor,
        p_start: torch.Tensor,
        p_end: torch.Tensor,
        num_samples: int = 32,
    ) -> torch.Tensor:
        interpolation = torch.linspace(
            0.0, 1.0, num_samples, device=volume.device
        )
        depth, height, width = volume.shape[-3:]
        points_z = (
            (p_start[0] + interpolation * (p_end[0] - p_start[0]))
            / (depth - 1)
            * 2.0
            - 1.0
        )
        points_y = (
            (p_start[1] + interpolation * (p_end[1] - p_start[1]))
            / (height - 1)
            * 2.0
            - 1.0
        )
        points_x = (
            (p_start[2] + interpolation * (p_end[2] - p_start[2]))
            / (width - 1)
            * 2.0
            - 1.0
        )
        grid = torch.stack([points_x, points_y, points_z], dim=1)
        grid = grid.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        if volume.ndim == 3:
            sampled = F.grid_sample(
                volume.unsqueeze(0).unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            return sampled.squeeze()
        if volume.ndim == 4:
            sampled = F.grid_sample(
                volume.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            return sampled[0, :, 0, 0, :].t()
        raise ValueError(f"Unexpected volume ndim: {volume.ndim}")

    def forward(
        self,
        seg_logits: torch.Tensor,
        keypoint_heatmaps: torch.Tensor,
        gt_keypoint_coords: torch.Tensor,
        gt_topo_coords: torch.Tensor,
        gt_branch_dirs: torch.Tensor,
        gt_fg_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, depth, height, width = seg_logits.shape
        device = seg_logits.device

        loss_keypoint = torch.tensor(0.0, device=device)
        valid_batches = 0
        for batch_index in range(batch_size):
            pred_coords = self._soft_argmax_3d(
                keypoint_heatmaps[batch_index],
                beta=1.0 / max(self.keypoint_sigma, 1e-3),
            )
            gt_coords = gt_keypoint_coords[batch_index]
            valid_mask = (
                (gt_coords[:, 0] >= 0)
                & (gt_coords[:, 1] >= 0)
                & (gt_coords[:, 2] >= 0)
            )
            if valid_mask.any():
                difference = pred_coords[valid_mask] - gt_coords[valid_mask]
                norm_factor = torch.tensor(
                    [depth, height, width], device=device, dtype=torch.float32
                )
                normalized_difference = difference / norm_factor.unsqueeze(0)
                loss_keypoint += (normalized_difference**2).mean()
                valid_batches += 1
        if valid_batches > 0:
            loss_keypoint = loss_keypoint / valid_batches

        loss_graph = torch.tensor(0.0, device=device)
        valid_edges_total = 0
        seg_probs = torch.softmax(seg_logits, dim=1)
        for batch_index in range(batch_size):
            gt_coords = gt_keypoint_coords[batch_index]
            pred_coords = self._soft_argmax_3d(
                keypoint_heatmaps[batch_index],
                beta=1.0 / max(self.keypoint_sigma, 1e-3),
            )
            for edge_i, edge_j in self.template_edges:
                gt_start = gt_coords[edge_i]
                gt_end = gt_coords[edge_j]
                if gt_start[0] < 0 or gt_end[0] < 0:
                    continue
                pred_start = pred_coords[edge_i]
                pred_end = pred_coords[edge_j]
                foreground_probability = seg_probs[batch_index, 1:].sum(dim=0)
                sampled = self._sample_line_3d(
                    foreground_probability,
                    pred_start,
                    pred_end,
                    self.edge_sample_points,
                )
                loss_graph += F.binary_cross_entropy(
                    sampled.clamp(1e-7, 1 - 1e-7), torch.ones_like(sampled)
                )
                valid_edges_total += 1
        if valid_edges_total > 0:
            loss_graph = loss_graph / valid_edges_total

        loss_branch_dir = torch.tensor(0.0, device=device)
        valid_voxels = 0
        for batch_index in range(batch_size):
            branch_id_map = gt_topo_coords[batch_index, ..., 0]
            foreground_voxels = branch_id_map >= 0
            if not foreground_voxels.any():
                continue
            foreground_indices = torch.nonzero(
                foreground_voxels, as_tuple=False
            )
            if len(foreground_indices) == 0:
                continue
            max_samples = 8192
            if len(foreground_indices) > max_samples:
                permutation = torch.randperm(
                    len(foreground_indices), device=device
                )[:max_samples]
                foreground_indices = foreground_indices[permutation]
            valid_voxels += len(foreground_indices)

            gt_direction = gt_branch_dirs[
                batch_index,
                foreground_indices[:, 0],
                foreground_indices[:, 1],
                foreground_indices[:, 2],
            ]
            gt_direction_normalized = gt_direction / (
                gt_direction.norm(dim=1, keepdim=True) + 1e-8
            )
            foreground_probability = seg_probs[batch_index, 1:].sum(dim=0)
            foreground_probability = foreground_probability.unsqueeze(0).unsqueeze(0)
            gradient_z = (
                foreground_probability[:, :, 2:, :, :]
                - foreground_probability[:, :, :-2, :, :]
            )
            gradient_y = (
                foreground_probability[:, :, :, 2:, :]
                - foreground_probability[:, :, :, :-2, :]
            )
            gradient_x = (
                foreground_probability[:, :, :, :, 2:]
                - foreground_probability[:, :, :, :, :-2]
            )
            gradient_z = F.pad(gradient_z, (0, 0, 0, 0, 1, 1))
            gradient_y = F.pad(gradient_y, (0, 0, 1, 1, 0, 0))
            gradient_x = F.pad(gradient_x, (1, 1, 0, 0, 0, 0))
            gradient_z = gradient_z.squeeze(0).squeeze(0)
            gradient_y = gradient_y.squeeze(0).squeeze(0)
            gradient_x = gradient_x.squeeze(0).squeeze(0)
            pred_gradient = torch.stack(
                [
                    gradient_z[
                        foreground_indices[:, 0],
                        foreground_indices[:, 1],
                        foreground_indices[:, 2],
                    ],
                    gradient_y[
                        foreground_indices[:, 0],
                        foreground_indices[:, 1],
                        foreground_indices[:, 2],
                    ],
                    gradient_x[
                        foreground_indices[:, 0],
                        foreground_indices[:, 1],
                        foreground_indices[:, 2],
                    ],
                ],
                dim=1,
            )
            pred_gradient_normalized = pred_gradient / (
                pred_gradient.norm(dim=1, keepdim=True) + 1e-8
            )
            cosine_similarity = (
                pred_gradient_normalized * gt_direction_normalized
            ).sum(dim=1)
            loss_branch_dir += (1.0 - cosine_similarity).mean()
        if valid_voxels > 0:
            loss_branch_dir = loss_branch_dir / batch_size

        total = (
            self.lambda_keypoint * loss_keypoint
            + self.lambda_graph * loss_graph
            + self.lambda_branch_dir * loss_branch_dir
        )
        return {
            "total": total,
            "loss_keypoint": loss_keypoint,
            "loss_graph": loss_graph,
            "loss_branch_dir": loss_branch_dir,
        }


class CombinedTopologyLoss(nn.Module):
    """Combine segmentation, clDice, and template graph topology losses."""

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
        seg_logits,
        labels,
        keypoint_heatmaps: Optional[torch.Tensor] = None,
        gt_keypoint_coords: Optional[torch.Tensor] = None,
        gt_topo_coords: Optional[torch.Tensor] = None,
        gt_branch_dirs: Optional[torch.Tensor] = None,
        voxel_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_seg = self.seg_loss(
            seg_logits, labels, voxel_weight=voxel_weight
        )
        total = loss_seg
        losses = {
            "loss_seg": loss_seg,
            "loss_cldice": torch.tensor(
                0.0,
                device=(
                    seg_logits.device
                    if isinstance(seg_logits, torch.Tensor)
                    else loss_seg.device
                ),
            ),
            "loss_keypoint": torch.tensor(0.0),
            "loss_graph": torch.tensor(0.0),
            "loss_branch_dir": torch.tensor(0.0),
        }
        device = loss_seg.device
        if self.lambda_cldice > 0 and labels is not None:
            primary_logits = (
                seg_logits
                if not isinstance(seg_logits, (list, tuple))
                else seg_logits[0]
            )
            seg_probs = torch.softmax(primary_logits, dim=1)
            if labels.ndim == 4:
                gt_binary = ((labels >= 1) & (labels <= 4)).float()
            else:
                gt_binary = labels[:, 1:].sum(dim=1).clamp(0, 1)
            loss_cldice = self.cldice_loss(seg_probs, gt_binary)
            total = total + self.lambda_cldice * loss_cldice
            losses["loss_cldice"] = loss_cldice.to(device)

        if keypoint_heatmaps is not None and gt_keypoint_coords is not None:
            primary_logits = (
                seg_logits
                if not isinstance(seg_logits, (list, tuple))
                else seg_logits[0]
            )
            graph_losses = self.graph_loss(
                primary_logits,
                keypoint_heatmaps,
                gt_keypoint_coords,
                gt_topo_coords if gt_topo_coords is not None else torch.zeros(1),
                gt_branch_dirs if gt_branch_dirs is not None else torch.zeros(1),
            )
            total = total + graph_losses["total"]
            losses["loss_keypoint"] = graph_losses["loss_keypoint"].to(device)
            losses["loss_graph"] = graph_losses["loss_graph"].to(device)
            losses["loss_branch_dir"] = graph_losses["loss_branch_dir"].to(device)
        losses["total"] = total
        return losses


__all__ = [
    "SoftClDiceLoss",
    "TemplateGraphTopologyLoss",
    "CombinedTopologyLoss",
]
