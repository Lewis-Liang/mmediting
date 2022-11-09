import torch
from mmedit.models.backbones.sr_backbones.basicvsr_pp_unet import BasicVSRPlusPlusUnet


if __name__ == "__main__":
    # output has the same size as input
    model = BasicVSRPlusPlusUnet(
        mid_channels=64,
        is_low_res_input=False,
        spynet_pretrained=None,
        cpu_cache_length=100).cuda()
    input_tensor = torch.rand(1, 5, 3, 256, 256).cuda()
    output = model(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)
    