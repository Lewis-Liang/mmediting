exp_name = 'basicvsr_dehazenet_c64n10_300k_revide'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRDehazeNet',
        mid_channels=64,
        num_blocks=10,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    # 原损失：
    # pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
    
    # 自定义损失：
    # TEST1：L1Loss + percp
    # TEST2：L1Loss
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
# train_cfg参考了CVPR 2022论文参数设置
train_cfg = dict(fix_iter=2500)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'DehazeFolderMultipleGTDataset'
val_dataset_type = 'DehazeFolderMultipleGTDataset'
test_dataset_type = 'DehazeFolderMultipleGTDataset'

# RescaleToZeroOne：仅仅除以像素最大值归一化到[0,1]
# Normalize：给定mean和std（mmedit中是127.5，而不是0.5，因为不会事先归一化到0到1之间）
# Normalize([127.5]*3,[127.5]*3) = RescaleToZeroOne + Normalize([.5]*3,[.5]*3)

# img norm cfg
img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], to_rgb=True)

train_pipeline = [
    dict(type='GenerateSegmentIndices', 
         interval_list=[1],
         filename_tmpl='{:05d}.JPG'),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
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
    dict(type='Normalize', keys=['lq', 'gt'], **img_norm_cfg),
    dict(type='PairedRandomCropWithoutScale', gt_patch_size=384),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

val_pipeline = [
    dict(type='GenerateSegmentIndices', 
         interval_list=[1],
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
    dict(type='Normalize', keys=['lq', 'gt'], **img_norm_cfg),
    dict(type='PairedRandomCropWithoutScale', gt_patch_size=384),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', 
         interval_list=[1],
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
    dict(type='Normalize', keys=['lq', 'gt'], **img_norm_cfg),
    dict(
        type='Resize',
        keys=['lq','gt'],
        keep_ratio=True,
        scale=(1024, 2048),
        interpolation='bicubic'),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', 
         interval_list=[1],
         filename_tmpl='{:05d}.JPG'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='Normalize', keys=['lq'], **img_norm_cfg),
    dict(
        type='Resize',
        keys=['lq'],
        keep_ratio=True,
        scale=(1024, 2048),
        interpolation='bicubic'),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    # 1 gpus
    # workers_per_gpu从2变成4之后，GPU-Util利用率瞬间就上去了
    workers_per_gpu=4, 
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    # num_input_frames暂时调成5帧，并设置RepeatDataset
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
    # val和test的batch均为1，帧长也调整为10
    val=dict(
        type=val_dataset_type,
        lq_folder='./data/REVIDE_indoor/Test/hazy',
        gt_folder='./data/REVIDE_indoor/Test/gt',
        num_input_frames=5,
        pipeline=val_pipeline,
        test_mode=True),
    # test
    test=dict(
        type=test_dataset_type,
        lq_folder='./data/REVIDE_indoor/Test/hazy',
        gt_folder='./data/REVIDE_indoor/Test/gt',
        num_input_frames=5,
        pipeline=test_pipeline,
        test_mode=True),
)

# optimizer
# 参考了CVPR 2022论文的参数设置
optimizers = dict(
    generator=dict(
        type='AdamW',
        lr=2e-4,
        betas=(0.9, 0.999),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})))

# learning policy
# total_iters参考了CVPR 2022论文的参数设置
total_iters = 50000
# lr_cofig未修改
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[50000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=1000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
# save_image初值为False
evaluation = dict(interval=1000, save_image=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# train
# resume_from = None

# test
resume_from = 'work_dirs/basicvsr_dehazenet_c64n10_300k_revide/iter_50000.pth'
workflow = [('train', 1)]
work_dir = f'./work_dirs/{exp_name}'
find_unused_parameters = True

