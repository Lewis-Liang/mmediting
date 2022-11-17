# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import glob
import numbers
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

import mmcv
from mmcv.utils import Config
from mmcv.runner import HOOKS, Hook

from mmedit.datasets.pipelines import Compose
from .metrics import psnr, ssim
from ..misc import tensor2img


class EvalIterHook(Hook):
    """Non-Distributed evaluation hook for iteration-based runner.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval. Default: 1.
        eval_kwargs (dict): Other eval kwargs. It contains:
            save_image (bool): Whether to save image.
            save_path (str): The path to save image.
    """

    def __init__(self, dataloader, interval=1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, '
                            f'but got { type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.save_image = self.eval_kwargs.pop('save_image', False)
        self.save_path = self.eval_kwargs.pop('save_path', None)

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        from mmedit.apis import single_gpu_test
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            save_image=self.save_image,
            save_path=self.save_path,
            iteration=runner.iter)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Evaluation function.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
            results (dict): Model forward results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        # call `after_val_epoch` after evaluation.
        # This is a hack.
        # Because epoch does not naturally exist In IterBasedRunner,
        # thus we consider the end of an evluation as the end of an epoch.
        # With this hack , we can support epoch based hooks.
        if 'iter' in runner.__class__.__name__.lower():
            runner.call_hook('after_val_epoch')


class DistEvalIterHook(EvalIterHook):
    """Distributed evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval. Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        eval_kwargs (dict): Other eval kwargs. It may contain:
            save_image (bool): Whether save image.
            save_path (str): The path to save image.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 **eval_kwargs):
        super().__init__(dataloader, interval, **eval_kwargs)
        self.gpu_collect = gpu_collect

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        from mmedit.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect,
            save_image=self.save_image,
            save_path=self.save_path,
            iteration=runner.iter)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)


allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

def evaluate(output, gt, metrics=['PSNR', 'SSIM'], crop_border=0):
    global allowed_metrics
    output = tensor2img(output)
    gt = tensor2img(gt)
    eval_result = dict()
    for metric in metrics:
        eval_result[metric] = allowed_metrics[metric](output, gt, crop_border)
    return eval_result

def restoration_video_inference(model,
                                                                    img_dir,
                                                                    start_idx,
                                                                    filename_tmpl,
                                                                    max_seq_len=None):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img_dir (str): Directory of the input video.
        start_idx (int): The index corresponds to the first frame in the
            sequence.
        filename_tmpl (str): Template for file name.
        max_seq_len (int | None): The maximum sequence length that the model
            processes. If the sequence length is larger than this number,
            the sequence is split into multiple segments. If it is None,
            the entire sequence is processed at once.

    Returns:
        Tensor: The predicted restoration result.
    """
    
    # model
    model.eval()
    device = next(model.parameters()).device
    # backup model.test_cfg.metrics
    test_cfg_backup = model.cfg.test_cfg
    model.module.test_cfg = None

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # the first element in the pipeline must be 'GenerateSegmentIndices'
    if test_pipeline[0]['type'] != 'GenerateSegmentIndices':
        raise TypeError('The first element in the pipeline must be '
                        f'"GenerateSegmentIndices", but got '
                        f'"{test_pipeline[0]["type"]}".')

    # specify start_idx and filename_tmpl
    test_pipeline[0]['start_idx'] = start_idx
    test_pipeline[0]['filename_tmpl'] = filename_tmpl

    # prepare data
    sequence_length = len(glob.glob(osp.join(img_dir, '*')))
    lq_folder = osp.dirname(img_dir)
    key = osp.basename(img_dir)
    data = dict(
        lq_path=lq_folder,
        gt_path='',
        key=key,
        sequence_length=sequence_length)

    # compose the pipeline
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = data['lq'].unsqueeze(0)  # in cpu

    # forward the model
    with torch.no_grad():
        if max_seq_len is None:
            result = model(
                lq=data.to(device), test_mode=True)['output'].cpu()
        else:
            result = []
            for i in range(0, data.size(1), max_seq_len):
                # 此处model实际调用的参数包括lq,test_mode两个，test_mode用于控制BaseRestorer的forward方向(train或test)
                result.append(
                    model(
                        lq=data[:, i:i + max_seq_len].to(device),
                        test_mode=True)['output'].cpu())
            result = torch.cat(result, dim=1)
            
    # recover test_cfg
    model.module.test_cfg = test_cfg_backup
    
    return result


@HOOKS.register_module() 
class EvalIterHookFull(Hook):
    """使用验证视频中所有的数据，而不是只使用采样的几帧
          并且保存最优psnr和ssim的模型
    Args:
        interval (int): Evaluation interval. Default: 1.
        eval_kwargs (dict): Other eval kwargs. It contains:
            save_image (bool): Whether to save image.
            save_path (str): The path to save image.
    """

    def __init__(self, config, interval=1, **eval_kwargs):
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_iter = 0

        self.config = config
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.save_image = self.eval_kwargs.pop('save_image', False)
        self.save_path = self.eval_kwargs.pop('save_path', None)

    def after_train_iter(self, runner):
        assert self.config is not None, "config file path should be assigned in config file"
        if not self.every_n_iters(runner, self.interval):
            return
        
        runner.log_buffer.clear()
     
        cfg = Config.fromfile(self.config)
        test_folder = cfg.data.test.lq_folder
        test_folders = os.listdir(test_folder)
        prog_bar = mmcv.ProgressBar(len(test_folders))
        
        start_idx=0
        filename_tmpl='{:05d}.JPG'
        filename_suffix=osp.splitext(filename_tmpl)[1]
        evaluate_results = dict()
        
        for folder_name in test_folders:
            img_dir = osp.join(test_folder, folder_name)
            iteration = runner.iter
            key = osp.basename(img_dir)
            save_path = cfg.work_dir if self.save_path is None else self.save_path
            sequence_length = len(glob.glob(osp.join(img_dir, '*')))

            # prepare Output data 
            if torch.cuda.is_available():
                max_seq_len = cfg.data.val.num_input_frames
                runner.model.cfg = cfg
                output = restoration_video_inference(runner.model, img_dir,  start_idx, filename_tmpl, max_seq_len)

                # prepare Ground Truth data
                lq_folder = cfg.data.test['lq_folder']
                gt_folder = cfg.data.test['gt_folder']
                test_pipeline = cfg.test_pipeline
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
                if self.save_image:
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
        self.evaluate(runner, evaluate_results)

    def evaluate(self, runner, results):
        """Evaluation function.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
            results (dict): Model forward results.
        """
        eval_res = defaultdict(list)  # a dict of list

        for _, res in results.items():
            for metric, val in res.items():
                eval_res[metric].append(val)

        # average the results
        eval_res = {
            metric: sum(values) / len(values)
            for metric, values in eval_res.items()
        }
        # compare best model
        cur_psnr = eval_res['PSNR']
        cur_ssim = eval_res['SSIM']
        if cur_psnr > self.best_psnr:
            from mmcv.runner.checkpoint import save_checkpoint
            self.best_psnr = cur_psnr
            self.best_ssim = cur_ssim
            self.best_iter = runner.iter
            # log best
            runner.log_buffer.output['best_psnr'] = self.best_psnr
            runner.log_buffer.output['best_ssim'] = self.best_ssim
            # record best
            with open(osp.join(runner.work_dir, "best.txt"), "w") as best_recorder:
                best_recorder.write(f'best_iter:{self.best_iter}\
                                                            \nbest_psnr:{self.best_psnr}\
                                                            \nbest_ssim:{self.best_ssim}')
            # save chekpoint
            runner.meta.update(epoch=runner.epoch + 1, iter=runner.iter)
            save_checkpoint(runner.model, 
                                                filename=osp.join(runner.work_dir, 'best.pth'), 
                                                optimizer=runner.optimizer, 
                                                meta=runner.meta)
        
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        # call `after_val_epoch` after evaluation.
        # This is a hack.
        # Because epoch does not naturally exist In IterBasedRunner,
        # thus we consider the end of an evluation as the end of an epoch.
        # With this hack , we can support epoch based hooks.
        if 'iter' in runner.__class__.__name__.lower():
            runner.call_hook('after_val_epoch')

