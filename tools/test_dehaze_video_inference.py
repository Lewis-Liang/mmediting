import torch

from mmedit.apis import init_model, restoration_video_inference


def test_restoration_video_inference():
    if torch.cuda.is_available():
        # recurrent framework (BasicVSR)
        model = init_model(
            './configs/restorers/basicvsr_dehaze/basicvsr_dehaze_c64n10_300k_revide.py',
            None,
            device='cuda')
        img_dir = './data/REVIDE_indoor/Test/hazy/W002'
        window_size = 0
        start_idx = 0
        filename_tmpl = '{:05d}.JPG'

        output = restoration_video_inference(model, img_dir, window_size,
                                             start_idx, filename_tmpl)
        print(output.shape)
        
if __name__ == '__main__':
    test_restoration_video_inference()

