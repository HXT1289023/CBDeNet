import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.structures import InstanceData
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.utils import multi_apply
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.structures.bbox import HorizontalBoxes

from ..layers.utils import multiclass_nms
from ..layers.custom_modules import Conv, DFL


@MODELS.register_module()
class EfficientDecoder(BaseDenseHead):
    """Anchor-free decoder with SimOTA assignment and DFL regression."""

    def __init__(
        self,
        num_classes,
        in_channels,
        reg_max=16,
        train_cfg=None,
        test_cfg=None,
        loss_cls=dict(type='DABLoss', loss_weight=1.0),
        loss_bbox=dict(type='CIoULoss', loss_weight=2.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.num_classes = int(num_classes)
        self.in_channels = in_channels
        self.reg_max = int(reg_max)  # 语义：最大距离桶上界
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.prior_generator = MlvlPointGenerator([8, 16, 32])  # P3/P4/P5
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dfl = MODELS.build(loss_dfl)

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.sampler = TASK_UTILS.build(self.train_cfg.sampler, default_args=dict(context=self))

        self._init_layers()

    def _init_layers(self):
        R = self.reg_max + 1  # 统一：桶数 = reg_max + 1
        self.stems = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for c in self.in_channels:
            self.stems.append(nn.Sequential(
                Conv(c, c, 3, g=max(1, c // 16)),
                Conv(c, c, 3, g=max(1, c // 16)),
            ))
            # 回归分支通道数 = 4 * (reg_max + 1)
            self.reg_convs.append(nn.Conv2d(c, 4 * R, 1))
            # 分类分支
            cls = nn.Conv2d(c, self.num_classes, 1)
            # 初始分类先验：约 1%
            if cls.bias is not None:
                nn.init.constant_(cls.bias, -4.595)
            self.cls_convs.append(cls)
        # DFL 模块（这里保留占位，不直接在 forward 用）
        self.dfl = DFL(self.reg_max) if self.reg_max > 0 else nn.Identity()

    def init_weights(self):
        super().init_weights()
        # 回归分支 bias 初始化
        for reg_conv in self.reg_convs:
            if hasattr(reg_conv, 'bias') and reg_conv.bias is not None:
                reg_conv.bias.data[:] = 1.0

    def forward(self, feats):
        cls_scores, reg_dist_preds = multi_apply(
            self.forward_single, feats, self.stems, self.reg_convs, self.cls_convs
        )
        return cls_scores, reg_dist_preds

    @staticmethod
    def forward_single(x, stem, reg_conv, cls_conv):
        x = stem(x)
        reg_dist_pred = reg_conv(x)      # (B, 4*(R), H, W) 其中 R=reg_max+1
        cls_score = cls_conv(x)          # (B, C, H, W)
        return cls_score, reg_dist_pred

    # ----------------------- 训练损失 -----------------------
    def loss_by_feat(
        self,
        cls_scores,
        reg_dist_preds,
        batch_gt_instances,
        batch_img_metas,
        batch_gt_instances_ignore=None
    ):
        device = cls_scores[0].device
        num_imgs = len(batch_img_metas)
        R = self.reg_max + 1  # 统一用 R

        # 1) priors(像素) & strides
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_priors_pix = self.prior_generator.grid_priors(featmap_sizes, device=device)  # list[(Ni,2)]
        priors_pix = torch.cat(mlvl_priors_pix, dim=0).to(device)  # (sumN, 2)
        strides_tensor = torch.cat([
            torch.full((len(p), 1), stride[0], device=device, dtype=torch.float32)
            for p, stride in zip(mlvl_priors_pix, self.prior_generator.strides)
        ], dim=0)  # (sumN, 1)
        sumN = priors_pix.size(0)

        # 2) 展平预测
        flatten_cls = torch.cat(
            [cs.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cs in cls_scores],
            dim=1
        )  # (B, sumN, C)
        flatten_regd = torch.cat(
            [rd.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * R) for rd in reg_dist_preds],
            dim=1
        )  # (B, sumN, 4R)

        # 3) 解码预测框（像素）
        priors_pix_rep  = priors_pix.unsqueeze(0).expand(num_imgs, sumN, 2).reshape(-1, 2)
        strides_rep     = strides_tensor.unsqueeze(0).expand(num_imgs, sumN, 1).reshape(-1, 1)
        pred_xywh_pix   = self._decode_bbox(
            priors_pix_rep, flatten_regd.reshape(-1, 4 * R), strides_rep
        ).reshape(num_imgs, sumN, 4)
        pred_xyxy_pix   = self._xywh_to_xyxy(pred_xywh_pix)

        # 4) 分配（给 assigner 的是 概率 & xyxy 像素坐标）
        per_img_cls_for_assign = [flatten_cls[i].detach().sigmoid() for i in range(num_imgs)]
        per_img_pred_xyxy      = [pred_xyxy_pix[i].detach() for i in range(num_imgs)]
        priors_pix_list = [priors_pix] * num_imgs
        strides_list    = [strides_tensor] * num_imgs

        (labels, label_weights, bbox_targets,
         pos_inds_list, neg_inds_list, _) = multi_apply(
            self._get_target_single,
            per_img_cls_for_assign,
            priors_pix_list,
            strides_list,
            per_img_pred_xyxy,
            batch_gt_instances
        )

        labels         = torch.cat(labels, 0)
        label_weights  = torch.cat(label_weights, 0)
        bbox_targets   = torch.cat(bbox_targets, 0)    # xyxy (像素)
        pos_inds       = torch.cat(pos_inds_list, 0)

        if pos_inds.numel() == 0:
            zero = flatten_cls.sum() * 0.0
            return dict(loss_cls=zero, loss_bbox=zero, loss_dfl=zero)

        # 5) 分类(QFL)：用 IoU 当质量（让梯度流向回归）
        pred_scores = flatten_cls.reshape(-1, self.num_classes)  # raw logits
        target_scores = pred_scores.new_zeros(pred_scores.shape)

        with torch.no_grad():
            pos_labels = labels[pos_inds].clamp_(0, self.num_classes - 1)
            pos_tgt_xyxy  = bbox_targets[pos_inds]

        # 用带梯度的预测框算 IoU
        pos_regd = flatten_regd.reshape(-1, 4 * R)[pos_inds]
        pos_priors_pix = priors_pix_rep[pos_inds]
        pos_strides = strides_rep[pos_inds]
        pos_pred_xywh_pix_grad = self._decode_bbox(pos_priors_pix, pos_regd, pos_strides)
        pos_pred_xyxy_grad = self._xywh_to_xyxy(pos_pred_xywh_pix_grad)
        iou_grad = self._bbox_iou(pos_pred_xyxy_grad, pos_tgt_xyxy.detach()).clamp_(0.0, 1.0)

        # —— QFL地板（C）：给正样本质量一个下限，帮助早期排序
        iou_grad = iou_grad.clamp_(min=0.2)

        target_scores[pos_inds, pos_labels] = iou_grad

        # 数值稳定
        target_scores = torch.nan_to_num(target_scores, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
        pred_scores = torch.nan_to_num(pred_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # QFL 归一化
        qfl_avg_factor = float(target_scores.sum().detach().clamp(min=1.0).item())
        loss_cls = self.loss_cls(pred_scores, target_scores, avg_factor=qfl_avg_factor)

        # 6) 回归(IoU/DIoU)：使用非梯度预测框监督
        pos_pred_xyxy_detached = pred_xyxy_pix.reshape(-1, 4)[pos_inds]
        num_pos_all = max(sum(len(p) for p in pos_inds_list), 1)
        loss_bbox = self.loss_bbox(pos_pred_xyxy_detached, pos_tgt_xyxy, avg_factor=num_pos_all)

        # 7) DFL：grid 尺度 (l,t,r,b)
        pos_regd = flatten_regd.reshape(-1, 4 * R)[pos_inds]   # (P, 4R)
        pos_priors_grid = (priors_pix_rep / strides_rep).reshape(-1, 2)[pos_inds]
        tgt_xyxy_grid   = (pos_tgt_xyxy / strides_rep.reshape(-1, 1)[pos_inds]).to(pos_priors_grid.dtype)
        bbox_targets_dist = self._bbox2dist(pos_priors_grid, tgt_xyxy_grid)  # (P,4)

        loss_dfl = self.loss_dfl(
            pos_regd.view(-1, R),
            bbox_targets_dist.view(-1),
            avg_factor=4.0 * num_pos_all
        )

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

    # ------- 单图分配 -------
    def _get_target_single(
        self,
        cls_preds,          # (sumN, C) 概率
        priors_pix,         # (sumN, 2) 像素坐标 (cx, cy)
        strides,            # (sumN, 1)
        pred_xyxy,          # (sumN, 4) 预测框，像素尺度 xyxy
        gt_instances        # InstanceData，含 .bboxes 与 .labels
    ):
        device = cls_preds.device
        sumN = priors_pix.size(0)

        num_gt = len(getattr(gt_instances, 'bboxes', []))
        if num_gt == 0:
            labels = priors_pix.new_full((sumN,), self.num_classes, dtype=torch.long)
            label_weights = priors_pix.new_zeros(sumN)
            bbox_targets = priors_pix.new_zeros((sumN, 4))
            empty_idx = priors_pix.new_zeros(0, dtype=torch.long)
            return labels, label_weights, bbox_targets, empty_idx, empty_idx, None

        scores_01 = cls_preds
        if scores_01.max() > 1 or scores_01.min() < 0:
            scores_01 = scores_01.sigmoid()

        if strides.dim() == 1:
            strides = strides.view(-1, 1)
        strides = strides.to(device=device, dtype=priors_pix.dtype)
        strides_xy = strides.repeat(1, 2)
        priors_with_stride = torch.cat([priors_pix, strides_xy], dim=1)    # (N,4)

        pred_instances = InstanceData()
        pred_instances.priors = priors_with_stride
        pred_instances.scores = scores_01
        pred_instances.bboxes = HorizontalBoxes(pred_xyxy)

        if isinstance(gt_instances.bboxes, torch.Tensor):
            gt_boxes_boxes = HorizontalBoxes(gt_instances.bboxes)  # sampler
            gt_boxes_tensor = gt_instances.bboxes                  # assigner
        else:
            gt_boxes_boxes = gt_instances.bboxes
            gt_boxes_tensor = gt_instances.bboxes.tensor

        gt_instances_for_assign = InstanceData()
        gt_instances_for_assign.bboxes = gt_boxes_tensor
        gt_instances_for_assign.labels = gt_instances.labels

        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances_for_assign
        )

        gt_instances_for_sample = gt_instances.clone()
        gt_instances_for_sample.bboxes = gt_boxes_boxes
        sampling_result = self.sampler.sample(
            assign_result, pred_instances, gt_instances_for_sample
        )

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = priors_pix.new_full((sumN,), self.num_classes, dtype=torch.long)
        label_weights = priors_pix.new_zeros(sumN)
        bbox_targets = priors_pix.new_zeros((sumN, 4))

        if pos_inds.numel() > 0:
            labels[pos_inds] = sampling_result.pos_gt_labels
            label_weights[pos_inds] = 1.0
            pos_gt_b = sampling_result.pos_gt_bboxes
            pos_gt_xyxy = pos_gt_b.tensor if hasattr(pos_gt_b, 'tensor') else pos_gt_b
            bbox_targets[pos_inds] = pos_gt_xyxy

        return labels, label_weights, bbox_targets, pos_inds, neg_inds, sampling_result

    # ----------------------- 推理 -----------------------
    def predict_by_feat(self,
                        cls_scores,
                        reg_dist_preds,
                        batch_img_metas,
                        cfg=None,
                        rescale=False,
                        with_nms=True):

        num_levels = len(cls_scores)
        device = cls_scores[0].device
        R = self.reg_max + 1

        featmap_sizes = [cs.shape[-2:] for cs in cls_scores]
        mlvl_priors_pix = self.prior_generator.grid_priors(featmap_sizes, device=device)
        priors_pix = torch.cat(mlvl_priors_pix, dim=0)  # (N,2) 像素
        strides = torch.cat([
            torch.full((len(p), 1), s[0], device=device, dtype=torch.float32)
            for p, s in zip(mlvl_priors_pix, self.prior_generator.strides)
        ], dim=0)  # (N,1)

        test_cfg = self.test_cfg if cfg is None else cfg
        score_thr = float(getattr(test_cfg, 'score_thr', 0.0))
        nms_pre = int(getattr(test_cfg, 'nms_pre', 2000))
        max_per_img = int(getattr(test_cfg, 'max_per_img', 300))

        results_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            cls_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            reg_list = [reg_dist_preds[i][img_id].detach() for i in range(num_levels)]

            # 分类得分
            scores = torch.cat(
                [x.permute(1, 2, 0).reshape(-1, self.num_classes) for x in cls_list],
                dim=0
            ).sigmoid()  # (N, C)

            # 回归 logits 展平（关键：按通道数 C 展平）
            regd = torch.cat(
                [y.permute(1, 2, 0).reshape(-1, y.shape[0]) for y in reg_list],
                dim=0
            )  # (N, 4*R)

            # 解码到像素
            xywh_pix = self._decode_bbox(priors_pix, regd, strides)  # (N,4)
            cxcy = xywh_pix[:, :2]
            wh   = xywh_pix[:, 2:]
            xyxy = torch.cat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=1)

            # 放宽“最小尺寸”过滤，防止把盒子全扔了
            keep_wh = (wh.min(dim=1).values > 0.0)
            if keep_wh.any():
                xyxy = xyxy[keep_wh]
                scores = scores[keep_wh]
                regd_kept = regd[keep_wh]
            else:
                empty = InstanceData(
                    bboxes=HorizontalBoxes(xyxy.new_zeros((0, 4))),
                    scores=xyxy.new_zeros((0,)),
                    labels=xyxy.new_zeros((0,), dtype=torch.long),
                )
                results_list.append(empty)
                continue

            # —— 质量重评分（A）：用回归分布熵置信度重排
            regd_logits = regd_kept.view(-1, 4, R)            # (M,4,R)
            regd_prob = regd_logits.softmax(dim=2)             # (M,4,R)
            entropy = -(regd_prob * (regd_prob.clamp_min(1e-12).log())).sum(dim=2)  # (M,4)
            entropy = entropy.mean(dim=1)                      # (M,)
            q = (1.0 - (entropy / math.log(R))).clamp(0.0, 1.0)  # (M,)
            scores = scores * q.unsqueeze(1)

            # —— 砍低分长尾（B）：NMS 前先做一次大 Top-K 裁剪
            conf_max = scores.max(dim=1).values
            K1 = min(3000, xyxy.size(0))
            if xyxy.size(0) > K1:
                _, idx1 = conf_max.topk(K1)
                xyxy = xyxy[idx1]
                scores = scores[idx1]
                conf_max = conf_max[idx1]

            # 若仍全部 <= score_thr，则保底取前 300 个继续走
            if (conf_max <= score_thr).all():
                K2 = min(300, xyxy.size(0))
                _, idx2 = conf_max.topk(K2)
                xyxy = xyxy[idx2]
                scores = scores[idx2]
                conf_max = conf_max[idx2]

            # 常规 nms_pre
            if xyxy.size(0) > nms_pre:
                max_per_box = scores.max(dim=1).values
                _, topk_inds = max_per_box.topk(nms_pre)
                xyxy = xyxy[topk_inds]
                scores = scores[topk_inds]

            dets, labels = multiclass_nms(
                xyxy, scores, score_thr=score_thr, nms_cfg=test_cfg.nms, max_num=max_per_img
            )

            results = InstanceData()
            if dets.numel() == 0:
                results.bboxes = HorizontalBoxes(xyxy.new_zeros((0, 4)))
                results.scores = xyxy.new_zeros((0,))
                results.labels = xyxy.new_zeros((0,), dtype=torch.long)
            else:
                results.bboxes = HorizontalBoxes(dets[:, :4])
                results.scores = dets[:, 4]
                results.labels = labels

                # rescale=True 时用 ori_shape 裁剪；否则用 img_shape
                if rescale:
                    sf = img_meta['scale_factor']
                    if not torch.is_tensor(sf):
                        sf = torch.tensor(sf, device=results.bboxes.tensor.device,
                                          dtype=results.bboxes.tensor.dtype)
                    if sf.numel() == 2:
                        sf = torch.tensor([sf[0], sf[1], sf[0], sf[1]],
                                          device=results.bboxes.tensor.device,
                                          dtype=results.bboxes.tensor.dtype)
                    results.bboxes = HorizontalBoxes(results.bboxes.tensor / sf)
                    h, w = img_meta['ori_shape'][:2]
                else:
                    h, w = img_meta['img_shape'][:2]
                results.bboxes.clip_((h, w))

            results_list.append(results)

        return results_list

    # ----------------------- 互逆的几何变换 -----------------------
    @staticmethod
    def _dist2bbox_local(dist_ltbr_grid: torch.Tensor, priors_cxcy_grid: torch.Tensor) -> torch.Tensor:
        l, t, r, b = dist_ltbr_grid.unbind(dim=-1)
        cx, cy = priors_cxcy_grid.unbind(dim=-1)
        w = l + r
        h = t + b
        return torch.stack([cx, cy, w, h], dim=-1)

    # 位于 EfficientDecoder 类中的最终修正函数
    def _decode_bbox(self, priors_xy, reg_dist_preds, strides_tensor):
        """
        将基于分布的回归预测（ltrb）应用于先验点来解码边界框。
        所有几何计算都在“网格单位”中完成，与 _bbox2dist 互为逆运算。
        返回 xywh（像素）。
        """
        R = self.reg_max + 1
        # 1) logits -> 概率分布 -> 期望距离（grid 单位）
        dist = reg_dist_preds.view(-1, 4, R).softmax(dim=2)
        proj = torch.arange(R, device=reg_dist_preds.device, dtype=torch.float32)
        ltrb_grid = torch.matmul(dist, proj)  # (l,t,r,b) in grid units

        # 2) priors 从像素到 grid
        priors_xy_grid = priors_xy / strides_tensor

        # 3) grid 单位下的几何
        cx_grid, cy_grid = priors_xy_grid.unbind(dim=-1)
        l_grid, t_grid, r_grid, b_grid = ltrb_grid.unbind(dim=-1)

        x1_grid = cx_grid - l_grid
        y1_grid = cy_grid - t_grid
        x2_grid = cx_grid + r_grid
        y2_grid = cy_grid + b_grid

        # 4) 回到像素
        xyxy_grid = torch.stack([x1_grid, y1_grid, x2_grid, y2_grid], dim=-1)
        xyxy_pix = xyxy_grid * strides_tensor

        # 5) xyxy -> xywh
        return self._xyxy_to_xywh(xyxy_pix)

    @staticmethod
    def _xywh_to_xyxy(xywh):
        x, y, w, h = xywh.unbind(-1)
        half_w = w * 0.5
        half_h = h * 0.5
        x1 = x - half_w
        y1 = y - half_h
        x2 = x + half_w
        y2 = y + half_h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _xyxy_to_xywh(xyxy):
        x1, y1, x2, y2 = xyxy.unbind(-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=0.)
        h = (y2 - y1).clamp(min=0.)
        return torch.stack([cx, cy, w, h], dim=-1)

    def _bbox2dist(self, priors_cxcy_grid: torch.Tensor, gt_xyxy_grid: torch.Tensor) -> torch.Tensor:
        cx = priors_cxcy_grid[:, 0]
        cy = priors_cxcy_grid[:, 1]
        x1, y1, x2, y2 = gt_xyxy_grid.unbind(-1)
        l = (cx - x1).clamp_(min=0)
        t = (cy - y1).clamp_(min=0)
        r = (x2 - cx).clamp_(min=0)
        b = (y2 - cy).clamp_(min=0)
        # 目标范围 [0, reg_max]
        max_val = float(self.reg_max) - 1e-6
        return torch.stack([l, t, r, b], dim=1).to(torch.float32).clamp_(0.0, max_val)

    @staticmethod
    def _bbox_iou(b1: torch.Tensor, b2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        x1 = torch.max(b1[:, 0], b2[:, 0])
        y1 = torch.max(b1[:, 1], b2[:, 1])
        x2 = torch.min(b1[:, 2], b2[:, 2])
        y2 = torch.min(b1[:, 3], b2[:, 3])
        inter = (x2 - x1).clamp_(min=0) * (y2 - y1).clamp_(min=0)
        area1 = (b1[:, 2] - b1[:, 0]).clamp_(min=0) * (b1[:, 3] - b1[:, 1]).clamp_(min=0)
        area2 = (b2[:, 2] - b2[:, 0]).clamp_(min=0) * (b2[:, 3] - b2[:, 1]).clamp_(min=0)
        union = area1 + area2 - inter + eps
        return inter / union
