_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc_neu.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model=dict(
    # backbone=dict(
    #     type='ResNet',
    #     depth=18,
    #     norm_eval=False,
    #     norm_cfg=dict(type='BN'),
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,#残差模块组
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,#固定参数，使用预训练需要固定第一层
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        #norm_eval=False,
        #norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512],), 
    roi_head=dict(
        bbox_head=dict(num_classes=6)
    )
)

# optimizer = dict( 
#     _delete_=True, 
#     type='AdamW', 
#     lr=0.0001, 
#     weight_decay=0.1, 
#     paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[24, 33])
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, 33]) 
runner = dict(type='EpochBasedRunner', max_epochs=36)
#workflow = [('train', 1),('val',1)]
workflow=[('train', 1)]

work_dir = 'work_dirs/faster-res18' 
#python tools/train.py configs/det-code/faster-res18.py --seed 1172629384 --deterministic --seed 1775259909  150941787 
