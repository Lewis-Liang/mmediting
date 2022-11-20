# baseline in cvpr 2022 paper Neural Compression-Based Feature Learning for Video Restoration
# data aug: random horizontal, vertical, and transposed flipping
# fixiter 2500
# p 384
# b 16
# f 5
# lr 2e-4
# spynet_lr 2.5e-5
# adamw
# cosine annealing
# 50k

exp_name = 'basicvsr_plusplus_baseline_cvpr2022'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlus',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=False,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=2500)
test_cfg = dict(metrics=['PSNR','SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'DehazeFolderMultipleGTDataset'
val_dataset_type = 'DehazeFolderMultipleGTDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1],
        filename_tmpl='{:05d}.JPG'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Resize',
        keys=['lq','gt'],
        keep_ratio=False,
        scale=(1280, 720),
        interpolation='bicubic'),
    dict(type='PairedRandomCropWithoutScale', gt_patch_size=384),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1],filename_tmpl='{:05d}.JPG'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Resize',
        keys=['lq','gt'],
        keep_ratio=False,
        scale=(1280, 720),
        interpolation='bicubic'),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1],filename_tmpl='{:05d}.JPG'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(
        type='Resize',
        keys=['lq'],
        keep_ratio=False,
        scale=(1280, 720),
        interpolation='bicubic'),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),  # 8 gpus
    val_dataloader=dict(samples_per_gpu=1,workers_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='./data/REVIDE_indoor/Train/hazy',
            gt_folder='./data/REVIDE_indoor/Train/gt',
            num_input_frames=5,
            pipeline=train_pipeline,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='./data/REVIDE_indoor/Test/hazy',
        gt_folder='./data/REVIDE_indoor/Test/gt',
        pipeline=test_pipeline,
        num_input_frames=5,
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='./data/REVIDE_indoor/Test/hazy',
        gt_folder='./data/REVIDE_indoor/Test/gt',
        pipeline=test_pipeline,
        # num_input_frames=5,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='AdamW',
        lr=2e-4,
        betas=(0.9, 0.999),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})))

optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=8)

# learning policy
total_iters = 50000
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
# evaluation = dict(interval=1000, save_image=False)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# custom_hooks
custom_hooks = [ 
    dict(type='EvalIterHookFull', 
             config='configs/restorers/basicvsr_plusplus_dehaze/basicvsr_plusplus_baseline_cvpr2022.py', 
             interval=1000, 
             save_image=True,
             priority='LOW') 
]  

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True

