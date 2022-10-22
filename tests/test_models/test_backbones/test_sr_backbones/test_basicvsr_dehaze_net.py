import torch
from mmedit.models.backbones.sr_backbones.basicvsr_dehaze_net import BasicVSRDehazeNet


if __name__ == "__main__":
    # output has the same size as input
    model = BasicVSRDehazeNet(
        mid_channels=64,
        num_blocks=10,
        spynet_pretrained=None).cuda()
    input_tensor = torch.rand(1, 5, 3, 256, 256).cuda()
    output = model(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)
    print(output.shape)
    
    