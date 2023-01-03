import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

from ..registry import LOSSES
from .pixelwise_loss import L1Loss


def DarkChannel(im,sz=15):
    B, c, h, w = im.shape
    y = torch.zeros(1,1,h,w).cuda()
    for i in range(B):
        img = im[i].mul(255).byte()
        img = img.cpu().numpy().transpose((1, 2, 0))
        b, g, r = cv2.split(img)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark = cv2.erode(dc, kernel)
        out = torch.from_numpy(dark).cuda()
        out = out.float().div(255).unsqueeze(0).unsqueeze(0)
        y = torch.cat([y, out], dim=0)
    return y[1:]


def get_local_weights(residual, ksize):

    residual = residual.squeeze(1)
    pad = (ksize - 1) // 2
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')

    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight


def get_refined_artifact_map(img_gt, img_output, img_ema, ksize):
    b, t, c, h, w = img_gt.size()
    img_gt = img_gt.view(b*t, c, h, w)
    img_ema = img_ema.view(b*t, c, h, w)
    img_output = img_output.view(b*t, c, h, w)
    
    residual_ema = torch.sum(torch.abs(DarkChannel(img_gt).view(b,t,-1,h,w) - DarkChannel(img_ema).view(b,t,-1,h,w)), 1, keepdim=True)
    residual_SR = torch.sum(torch.abs(DarkChannel(img_gt).view(b,t,-1,h,w) - DarkChannel(img_output).view(b,t,-1,h,w)), 1, keepdim=True)

    patch_level_weight = torch.var(residual_SR.clone(), dim=(-1, -2, -3), keepdim=True) ** (1/5)
    pixel_level_weight = get_local_weights(residual_SR.clone(), ksize)
    overall_weight = patch_level_weight * pixel_level_weight

    overall_weight[residual_SR < residual_ema] = 0

    return overall_weight


# TODO with_ema?
@LOSSES.register_module()
class DCPLoss(nn.Module):
    def __init__(self, window_size=7, ldl_weight=1.0, with_ema=True):
        super().__init__()
        self.cri_artifacts = L1Loss()
        self.window_size = window_size
        self.ldl_weight = ldl_weight
        self.with_ema = with_ema
        
    def forward(self, gt, output, output_ema):
        pixel_weight = get_refined_artifact_map(gt, output, output_ema, self.window_size)
        l_g_artifacts = self.cri_artifacts(torch.mul(pixel_weight, output), torch.mul(pixel_weight, gt))
        return self.ldl_weight * l_g_artifacts

