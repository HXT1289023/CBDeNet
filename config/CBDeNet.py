import os
import sys

project_root = '/root/workspace/CBDeNet'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

custom_imports = dict(imports=['models'], allow_failed_imports=False)

_base_ = [
    '/root/workspace/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '/root/workspace/mmdetection/configs/_base_/default_runtime.py'
]

data_root = '/root/workspace/SCB-COCO/'
class_names = (
    'hand-raising',
    'reading',
    'writing',
    'using phone',
    'bowing the head',
    'leaning over the table',
)
num_classes = 6
metainfo = dict(classes=class_names)

deepen_factor = 0.5
widen_factor = 0.25
backbone_out_channels = [64, 128, 128, 256]
neck_out_channels = [64, 128, 256]

model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='HFEBackbone',
        in_channels=3,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        max_channels=1024,
        out_indices=(2, 4, 6, 10)),
    neck=dict(
        type='HiFusionFPN',
        in_channels=backbone_out_channels,
        out_channels=neck_out_channels,
        num_blocks=1,
        g=4),
    bbox_head=dict(
        type='EfficientDecoder',
        num_classes=num_classes,
        in_channels=neck_out_channels,
        reg_max=16,
        loss_cls=dict(type='DABLoss', loss_weight=1.0),
        loss_bbox=dict(type='CIoULoss', loss_weight=2.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25)),
    train_cfg=dict(
        assigner=dict(
            type='SimOTAAssigner',
            center_radius=2.5,
            iou_calculator=dict(type='BboxOverlaps2D')),
        sampler=dict(type='PseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        score_thr=1e-4,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=300))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=10.0, norm_type=2))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='SCB-Train/train.json',
        data_prefix=dict(img='SCB-Train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='SCB-Val/val.json',
        data_prefix=dict(img='SCB-Val/images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'SCB-Val/val.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator

