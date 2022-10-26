import numbers
import os
import os.path as osp
import torch
import mmcv
from mmedit.core import tensor2img
from mmedit.apis import init_model, restoration_video_inference


def test_restoration_video_inference(model, img_dir):
    if torch.cuda.is_available():
        # recurrent framework (BasicVSR)
        window_size = 0
        start_idx = 0
        filename_tmpl = '{:05d}.JPG'

        # max_seq_len表示一次最多输入和推理5帧
        output = restoration_video_inference(model, img_dir, window_size,
                                             start_idx, filename_tmpl, max_seq_len=5)
        return output
        
if __name__ == '__main__':
    test_folder = './data/REVIDE_indoor/Test/hazy'
    for folder_name in os.listdir(test_folder):
        img_dir = test_folder + '/' + folder_name
        save_path = 'outputs'
        iteration = None
        model = init_model(
            'configs/restorers/basicvsr_dehaze/basicvsr_dehaze_c64n10_300k_revide.py',
            'work_dirs/basicvsr_dehazenet_c64n10_300k_revide/iter_50000.pth',
            device='cuda')
        
        output = test_restoration_video_inference(model, img_dir)
        for i in range(0, output.size(1)):
            if isinstance(iteration, numbers.Number):
                save_path_i = osp.join(
                    save_path, folder_name,
                    f'{i:08d}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path_i = osp.join(save_path, folder_name,
                                        f'{i:08d}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                    f'but got {type(iteration)}')
            mmcv.imwrite(
                tensor2img(output[:, i, :, :, :]), save_path_i)

