exp_name = 'basicvsr_l1_vgg_ssim_accum_c64n10_1x1_50k_revide'

# model settings
model = dict(
    type='BasicVSR_vggloss',
    generator=dict(
        type='BasicVSRDehazeNet',
        is_low_res_input=False,
        mid_channels=64,
        num_blocks=30,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
    type='PerceptualLoss',
    layer_weights={
        '2': 0.1,
        '7': 0.1,
        '16': 1.0,
        '25': 1.0,
        '34': 1.0,
    },
    vgg_type='vgg19',
    perceptual_weight=0.1,
    style_weight=0,
    norm_img=False),
    )
# model training and testing settings
train_cfg = dict(fix_iter=5000)
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
    dict(
        type='Resize',
        keys=['lq','gt'],
        keep_ratio=False,
        scale=(1280, 720),
        interpolation='bicubic'),
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
    dict(type='GenerateSegmentIndices', 
         interval_list=[1],
         filename_tmpl='{:05d}.JPG'),
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
    # 1 gpus
    workers_per_gpu=2, 
    train_dataloader=dict(samples_per_gpu=1, drop_last=False),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='./data/REVIDE_indoor/Train/hazy',
            gt_folder='./data/REVIDE_indoor/Train/gt',
            num_input_frames=9,
            pipeline=train_pipeline,
            test_mode=False)),
    # val
    # val和test的batch均为1，帧长也调整为10
    val=dict(
        type=val_dataset_type,
        lq_folder='./data/REVIDE_indoor/Test/hazy',
        gt_folder='./data/REVIDE_indoor/Test/gt',
        num_input_frames=9,
        pipeline=val_pipeline,
        test_mode=True),
    # test
    test=dict(
        type=test_dataset_type,
        lq_folder='./data/REVIDE_indoor/Test/hazy',
        gt_folder='./data/REVIDE_indoor/Test/gt',
        num_input_frames=9,
        pipeline=test_pipeline,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=2e-4,
        betas=(0.9, 0.999),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})))
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=8
)

# learning policy
total_iters = 50000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[50000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=1000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
# save_image初值为False
evaluation = dict(interval=1000, save_image=False)
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
resume_from = None
workflow = [('train', 1)]
work_dir = f'./work_dirs/{exp_name}'
find_unused_parameters = True

