import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
from mmcv.runner import load_checkpoint
from torch.nn import functional as F
from torchvision import models

from .perceptual_loss import PerceptualVGG
from ..registry import LOSSES
from .perceptual_loss import PerceptualVGG


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


@LOSSES.register_module()
class ContrastLoss(nn.Module):
    def __init__(self, ablation=False, neg_num=1):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
        self.neg_num = neg_num

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                # n_vgg[i] [neg_num, c, h ,w]
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss


# TODO 暂时只实现了1:1的contrast loss
@LOSSES.register_module()
class ContrastLossWithPerceptualVGG(nn.Module):
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
                 ablation=False,
                 neg_num=1):
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
            
        self.neg_num = neg_num
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
        anchor_features = list(self.vgg(anchor).values())
        pos_features = list(self.vgg(pos.detach()).values())
        neg_features = list(self.vgg(neg.detach()).values())

        # calculate contrast loss
        contrast_loss = 0.0
        d_ap, d_an = 0, 0
        if self.contrast_weight > 0:
            for k in range(len(pos_features)):
                d_ap = self.criterion(anchor_features[k], pos_features[k])
                if not self.ab:
                    d_an = self.criterion(anchor_features[k], neg_features[k].detach())
                    contrastive = d_ap / (d_an + 1e-7)
                else:
                    contrastive = d_ap
                contrast_loss += contrastive * self.layer_weights[k]
            contrast_loss *= self.contrast_weight
        else:
            contrast_loss = None

        return contrast_loss


@LOSSES.register_module()
class ContrastLossTest(nn.Module):
    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 contrast_weight=1.0,
                 norm_img=True,
                 pretrained='torchvision://vgg19',
                 criterion='l1'):
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

    def forward(self, x, gt,neg):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.contrast_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.contrast_weight
        else:
            percep_loss = None

        return percep_loss
