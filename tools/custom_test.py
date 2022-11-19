# Copyright (c) OpenMMLab. All rights reserved.
from os.path import basename, splitext, exists

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils.path import fopen

from mmedit.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model
from mmedit.utils import setup_multi_processes

from tools.test import parse_args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **loader_cfg)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    args.save_image = args.save_path is not None
    empty_cache = cfg.get('empty_cache', False)
    if not distributed:
        _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model,
            data_loader,
            save_path=args.save_path,
            save_image=args.save_image)
    else:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = DistributedDataParallelWrapper(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

        device_id = torch.cuda.current_device()
        _ = load_checkpoint(
            model,
            args.checkpoint,
            map_location=lambda storage, loc: storage.cuda(device_id))
        outputs = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            save_path=args.save_path,
            save_image=args.save_image,
            empty_cache=empty_cache)

    # 保存eval_results到txt
    eval_results_txt = f'{cfg.work_dir}/eval_results.txt'
    if exists(eval_results_txt):
        handler =  fopen(eval_results_txt, mode="a")
    else:
        handler =  fopen(eval_results_txt, mode="w")
        handler.write(f'Exp_name: {cfg.exp_name}\n')
        
    # TODO model的meta属性内应该保存了iter信息
    checkpoint_name = splitext(basename(args.checkpoint))[0]
    if 'iter_' in checkpoint_name:
        checkpoint_iter = checkpoint_name[5:]
    else:
        checkpoint_iter = checkpoint_name
        
    # write iter to file
    handler.write('#'*10 + '\n')
    handler.write(f'Iter: {checkpoint_iter}\n')
    handler.write('#'*10 + '\n')
    
    if rank == 0 and 'eval_result' in outputs[0]:
        print('')
        # collect metrics
        msg_all = ''
        # print metrics
        stats = dataset.evaluate(outputs)
        for stat in stats:
            msg = 'Eval-{}: {}'.format(stat, stats[stat])
            msg_all += ( msg+'\n' )
            print(msg)
        # write metrics to file
        handler.write(msg_all)

        # save result pickle
        if args.out:
            print('writing results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
    
    handler.close()


if __name__ == '__main__':
    main()

# 保存到work_dirs/{exp_name}/test_result.csv下