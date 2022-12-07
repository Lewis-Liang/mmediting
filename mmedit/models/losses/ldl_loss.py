import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .pixelwise_loss import L1Loss


def get_local_weights(residual, ksize):

    pad = (ksize - 1) // 2
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')

    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight

def get_refined_artifact_map(img_gt, img_output, img_ema, ksize):

    residual_ema = torch.sum(torch.abs(img_gt - img_ema), 1, keepdim=True)
    residual_SR = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)

    patch_level_weight = torch.var(residual_SR.clone(), dim=(-1, -2, -3), keepdim=True) ** (1/5)
    pixel_level_weight = get_local_weights(residual_SR.clone(), ksize)
    overall_weight = patch_level_weight * pixel_level_weight

    overall_weight[residual_SR < residual_ema] = 0

    return overall_weight


@LOSSES.register_module()
class LDLLoss(nn.Module):
    def __init__(self, window_size=7, ldl_weight=1.0):
        super().__init__()
        self.cri_artifacts = L1Loss()
        self.window_size = window_size
        self.ldl_weight = ldl_weight
        
    def forward(self, gt, output, output_ema):
        pixel_weight = get_refined_artifact_map(gt, output, output_ema, self.window_size)
        l_g_artifacts = self.cri_artifacts(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
        return self.ldl_weight * l_g_artifacts

