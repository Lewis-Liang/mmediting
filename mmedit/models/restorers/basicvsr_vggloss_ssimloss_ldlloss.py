# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper

from ..registry import MODELS
from ..builder import  build_loss
from .basicvsr_vggloss_ssimloss import BasicVSR_vggloss_ssimloss


@MODELS.register_module()
class BasicVSR_vggloss_ssimloss_ldlloss(BasicVSR_vggloss_ssimloss):
    """BasicVSR model for video super-resolution.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        perceptual_loss (dict): Config for perceptual loss.
        ssim_loss (dict): Config for ssim loss.
        ldl_loss (dict): Config for ldl loss. is_use_ema=True is required
        ensemble (dict): Config for ensemble. Default: None.
        is_use_ema (bool, optional): When to apply exponential moving average
            on the network weights. Default: False.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 perceptual_loss=None,
                 ssim_loss = None,
                 ldl_loss = None,
                 ensemble=None,
                 is_use_ema=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(). __init__(generator, pixel_loss, perceptual_loss, ssim_loss,
                 ensemble,
                 is_use_ema,
                 train_cfg,
                 test_cfg,
                 pretrained)
        assert (ldl_loss is None) or (ldl_loss and self.is_use_ema==True), "ldl loss requires ema"
        # ldl_loss
        self.ldl_loss = build_loss(ldl_loss)


    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """
        
        losses = dict()
        output = self.generator(lq)
        loss_pix = self.pixel_loss(output, gt)
        losses['loss_pix'] = loss_pix
        
        # data
        gt_percep = gt.clone()
        gt_ssim = gt.clone()
        output_percep = output.clone()
        output_ssim = output.clone()
        
        # perceptual loss
        # reshape: (n, t, c, h, w) -> (n*t, c, h, w)
        c, h, w = gt.shape[2:]
        gt_percep = gt_percep.view(-1, c, h, w)
        output_percep = output_percep.view(-1, c, h, w)
        if self.perceptual_loss:
            loss_percep, loss_style = self.perceptual_loss(output_percep, gt_percep)
            if loss_percep is not None:
                losses['loss_perceptual'] = loss_percep
            if loss_style is not None:
                losses['loss_style'] = loss_style
                
        # ssim loss
        c, h, w = gt.shape[2:]
        gt_ssim = gt_ssim.view(-1, c, h, w)
        output_ssim = output_ssim.view(-1, c, h, w)
        if self.ssim_loss:
            loss_ssim = self.ssim_loss(output_ssim, gt_ssim)
            losses['loss_ssim'] = loss_ssim
            
        # ldl loss
        output_ema = self.generator_ema(lq)
        if self.ldl_loss:
            loss_ldl = self.ldl_loss(gt, output, output_ema)
            losses['loss_ldl'] = loss_ldl
                
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # during initialization, load weights from the ema model
        if (self.step_counter == self.start_iter
                and self.generator_ema is not None):
            if is_module_wrapper(self.generator):
                self.generator.module.load_state_dict(
                    self.generator_ema.module.state_dict())
            else:
                self.generator.load_state_dict(self.generator_ema.state_dict())
                
        # fix SPyNet and EDVR at the beginning
        if self.step_counter < self.fix_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if 'spynet' in k or 'edvr' in k:
                        v.requires_grad_(False)
        elif self.step_counter == self.fix_iter:
            # train all the parameters
            self.generator.requires_grad_(True)

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

