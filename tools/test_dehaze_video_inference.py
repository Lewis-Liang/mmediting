import argparse
import numbers
import os
import os.path as osp
import glob
import torch
import mmcv
from mmedit.core import psnr, ssim, tensor2img
from mmedit.apis import init_model, restoration_video_inference, restoration_video_inference_with_metrics
from mmcv.runner import load_checkpoint
from mmedit.models import build_model
from mmedit.datasets.pipelines import Compose


allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}


def init_model_with_metrics(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def evaluate(output, gt, metrics=['PSNR', 'SSIM'], crop_border=0):
    global allowed_metrics
    output = tensor2img(output)
    gt = tensor2img(gt)
    eval_result = dict()
    for metric in metrics:
        eval_result[metric] = allowed_metrics[metric](output, gt, crop_border)
    return eval_result

# recurrent framework (BasicVSR)
def test_restoration_video_inference(model, img_dir, window_size=0, start_idx=0, filename_tmpl='{:05d}.JPG', max_seq_len=5):
    if torch.cuda.is_available():
        # max_seq_len表示一次最多输入和推理的帧数
        output = restoration_video_inference(model, img_dir, window_size,
                                                          start_idx, filename_tmpl, max_seq_len)
        return output
    
    
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config to test')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint to test')
    parser.add_argument('--seq', type=int, required=True, help='max sequence length')
    parser.add_argument('--save-image', action='store_true', default=False, help='whether to save images')
    return parser


# TODO 写成一个argparse方便测试
if __name__ == '__main__':
    # REVIDE_indoor
    test_folder = './data/REVIDE_indoor/Test/hazy'
    # filename开始帧的数值
    start_idx=0
    filename_tmpl='{:05d}.JPG'
    filename_suffix=osp.splitext(filename_tmpl)[1]
    
    window_size=0
    evaluate_results = dict()
    
    parser = get_config()
    cfg = parser.parse_args()
    model = init_model(cfg.config, cfg.checkpoint, device='cuda')
    
    ############# 
    # Test begins
    #############
    test_folders = os.listdir(test_folder)
    prog_bar = mmcv.ProgressBar(len(test_folders))
    for folder_name in test_folders:
        img_dir = test_folder + '/' + folder_name
        iteration = None
        key = osp.basename(img_dir)
        save_path = 'outputs/' + model.cfg.exp_name
        sequence_length = len(glob.glob(osp.join(img_dir, '*')))

        # prepare Output data 
        output = test_restoration_video_inference(model, img_dir, max_seq_len=cfg.seq)

        # prepare Ground Truth data
        lq_folder = model.cfg.data.test['lq_folder']
        gt_folder = model.cfg.data.test['gt_folder']
        test_pipeline = model.cfg.test_pipeline
        test_pipeline[0]['start_idx'] = start_idx
        test_pipeline[0]['filename_tmpl'] = filename_tmpl
        data = dict(
            lq_path=lq_folder,
            gt_path=gt_folder,
            key=key,
            sequence_length=sequence_length)
        # compose the pipeline
        test_pipeline = Compose(test_pipeline)
        data = test_pipeline(data)
        gt = data['gt'].unsqueeze(0)  # in cpu

        # calculate metrics
        evaluate_results[img_dir] = evaluate(output, gt)
        
        # save output images
        if cfg.save_image:
            for i in range(0, output.size(1)):
                if isinstance(iteration, numbers.Number):
                    save_path_i = osp.join(
                        save_path, folder_name,
                        f'{i:08d}-{iteration + 1:06d}{filename_suffix}')
                elif iteration is None:
                    save_path_i = osp.join(save_path, folder_name,
                                            f'{i:08d}{filename_suffix}')
                else:
                    raise ValueError('iteration should be number or None, '
                                        f'but got {type(iteration)}')
                mmcv.imwrite(
                    tensor2img(output[:, i, :, :, :]), save_path_i)

        prog_bar.update()
    ############# 
    # Test ends
    #############

    # 平均每个img_dir的metrics
    metrics=['PSNR', 'SSIM']
    sum_metrics = dict( zip(metrics, [0., 0.]) )
    folders = os.listdir(test_folder)
    for metric in metrics:
        for folder_name in folders:
            img_dir = test_folder + '/' + folder_name
            sum_metrics[metric] += evaluate_results[img_dir][metric]
            
    
    for metric in metrics:
        print(f"{metric}:\t{sum_metrics[metric] / len(folders)}")

