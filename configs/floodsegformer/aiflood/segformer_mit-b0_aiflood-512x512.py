_base_ = [
    '../../_base_/models/segformer_mit-b0.py', '../../_base_/datasets/AIFloodSense.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_aiflood.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
    num_classes=4,
    use_werm=True,
    loss_decode=[
    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ]
    ))


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=470),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=470,
        end=4700,
        by_epoch=False,
    )
]
train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
