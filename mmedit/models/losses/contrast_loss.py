import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
from mmcv.runner import load_checkpoint
from torch.nn import functional as F

from ..registry import LOSSES


# TODO 暂时只实现了1:1的contrast loss
@LOSSES.register_module()
class ContrastLoss(nn.Module):
    """Contrast loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature for
            perceptual loss. Here is an example: {'4': 1., '9': 1., '18': 1.},
            which means the 5th, 10th and 18th feature layer will be
            extracted with weight 1.0 in calculating losses.
        layers_weights_style (dict): The weight for each layer of vgg feature
            for style loss. If set to 'None', the weights are set equal to
            the weights for perceptual loss. Default: None.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        contrast_weight (float): If `contrast_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'.
        criterion (str): Criterion type. Options are 'l1' and 'mse'.
            Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 contrast_weight=1.0,
                 norm_img=True,
                 pretrained='torchvision://vgg19',
                 criterion='l1',
                 ablation=False):
        super().__init__()
        self.norm_img = norm_img
        self.contrast_weight = contrast_weight
        self.layer_weights = layer_weights

        self.vgg = PerceptualVGG(
            layer_name_list=list(self.layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            pretrained=pretrained)

        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')
            
        self.ab = ablation

    def forward(self, anchor, pos, neg):
        """Forward function.

        Args:
            anchor (Tensor): Input tensor with shape (n, c, h, w).
            pos (Tensor): Ground-truth tensor with shape (n, c, h, w).
            neg (Tensor): Negative tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.norm_img:
            anchor = (anchor + 1.) * 0.5
            pos = (pos + 1.) * 0.5
            neg = (neg + 1.) * 0.5
        # extract vgg features
        anchor_features = self.vgg(anchor)
        pos_features = self.vgg(pos.detach())
        neg_features = self.vgg(neg.detach())

        # calculate contrast loss
        contrast_loss = 0.0
        d_ap, d_an = 0, 0
        if self.contrast_weight > 0:
            for k in anchor_features.keys():
                d_ap = self.criterion(anchor_features[k], pos_features[k])
                if not self.ab:
                    d_an = self.criterion(anchor_features[i], neg_features[i].detach())
                    contrastive = d_ap / (d_an + 1e-7)
                else:
                    contrastive = d_ap
                contrast_loss += contrastive * self.layer_weights[k]
            contrast_loss *= self.contrast_weight
        else:
            contrast_loss = None

        return contrast_loss

