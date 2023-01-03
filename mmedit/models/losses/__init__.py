# Copyright (c) OpenMMLab. All rights reserved.
from .composition_loss import (CharbonnierCompLoss, L1CompositionLoss,
                               MSECompositionLoss)
from .feature_loss import LightCNNFeatureLoss
from .gan_loss import DiscShiftLoss, GANLoss, GaussianBlur, GradientPenaltyLoss
from .gradient_loss import GradientLoss
from .perceptual_loss import (PerceptualLoss, PerceptualVGG,
                              TransferalPerceptualLoss)
from .pixelwise_loss import CharbonnierLoss, L1Loss, MaskedTVLoss, MSELoss
from .customed_loss import CharbonnierLoss_RGBandHSV
from .ssim_loss import SSIMLoss
from .ldl_loss import LDLLoss
from .dcp_loss import DCPLoss
from .contrast_loss import ContrastLoss,ContrastLossTest

from .utils import mask_reduce_loss, reduce_loss

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'L1CompositionLoss',
    'MSECompositionLoss', 'CharbonnierCompLoss', 'GANLoss', 'GaussianBlur',
    'GradientPenaltyLoss', 'PerceptualLoss', 'PerceptualVGG', 'reduce_loss',
    'mask_reduce_loss', 'DiscShiftLoss', 'MaskedTVLoss', 'GradientLoss',
    'TransferalPerceptualLoss', 'LightCNNFeatureLoss',
    'CharbonnierLoss_RGBandHSV', 'SSIMLoss', 'LDLLoss','ContrastLoss','ContrastLossTest'
]
