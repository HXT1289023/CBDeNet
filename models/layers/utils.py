# mmdet/models/liberrynet/layers/utils.py

import torch


# ---------------------- dist2bbox ----------------------
@torch.no_grad()
def dist2bbox(dist, priors, xywh=True):
    """
    Convert distances (l,t,r,b) from a center prior to bbox.
    Args:
        dist:   Tensor (N,4) = [l,t,r,b]  (in the same units as priors)
        priors: Tensor (N,2) = [cx,cy]
        xywh:   if True, return (cx,cy,w,h); else (x1,y1,x2,y2)
    """
    cxcy = priors
    lt = dist[:, 0:2]
    rb = dist[:, 2:4]
    x1y1 = cxcy - lt
    x2y2 = cxcy + rb
    if xywh:
        wh = (x2y2 - x1y1).clamp(min=0.0)
        ctr = (x1y1 + x2y2) * 0.5
        return torch.cat([ctr, wh], dim=1)
    else:
        return torch.cat([x1y1, x2y2], dim=1)


# ---------------------- make_anchors ----------------------
@torch.no_grad()
def make_anchors(feats, strides, grid_cell_offset: float = 0.5, device=None):
    """
    生成各层的网格中心(先验点) 与 对应 stride。
    兼容两种输入：
      1) feats: list[Tensor] 的特征图 (B,C,H,W)
      2) feats: list[tuple] / list[list] 只给 (H,W)

    返回：
      mlvl_priors:  list[Tensor], 每层形状为 (Hi*Wi, 2) ，单位为“grid 坐标”(cx, cy)
      mlvl_strides: list[Tensor], 每层形状为 (Hi*Wi, 1)
    """
    mlvl_priors = []
    mlvl_strides = []

    for i, fm in enumerate(feats):
        if isinstance(fm, (tuple, list)):
            H, W = int(fm[0]), int(fm[1])
        else:
            # 传入是真实特征图 (B,C,H,W)
            H, W = int(fm.shape[-2]), int(fm.shape[-1])

        s = strides[i][0] if isinstance(strides[i], (tuple, list)) else int(strides[i])
        dev = device if device is not None else (fm.device if torch.is_tensor(fm) else 'cpu')

        # 注意 meshgrid 新版需要 indexing='ij'
        yv, xv = torch.meshgrid(
            torch.arange(H, device=dev),
            torch.arange(W, device=dev),
            indexing='ij'
        )
        # grid 中心（加 0.5 的偏移）
        cx = (xv.to(torch.float32) + grid_cell_offset)
        cy = (yv.to(torch.float32) + grid_cell_offset)
        priors = torch.stack([cx, cy], dim=-1).reshape(-1, 2)  # (H*W, 2) grid 坐标

        stride_tensor = torch.full((H * W, 1), float(s), device=dev, dtype=torch.float32)

        mlvl_priors.append(priors)
        mlvl_strides.append(stride_tensor)

    return mlvl_priors, mlvl_strides


# ---------------------- NMS backends ----------------------
def _nms_mmcv(boxes, scores, iou_thr):
    """Try mmcv NMS, fallback if not available."""
    try:
        from mmcv.ops import nms as mmcv_nms  # mmcv>=2
        dets, keep = mmcv_nms(boxes, scores, iou_thr)
        return dets[:, :4], dets[:, 4], keep
    except Exception:
        return None


def _nms_torchvision(boxes, scores, iou_thr):
    from torchvision.ops import nms as tv_nms
    keep = tv_nms(boxes, scores, iou_thr)
    return boxes[keep], scores[keep], keep


def batched_nms(bboxes, scores, labels, nms_cfg):
    """
    A class-aware batched NMS.
    Args:
        bboxes: (M,4) in xyxy
        scores: (M,)
        labels: (M,) int64
        nms_cfg: dict like {'type':'nms','iou_threshold':0.5}
    Return:
        bboxes_keep: (K,4), scores_keep: (K,), keep_inds: (K,)
    """
    if bboxes.numel() == 0:
        return bboxes, scores, bboxes.new_zeros((0,), dtype=torch.long)

    iou_thr = float(nms_cfg.get('iou_threshold', 0.5))
    # 为不同类别加上一个大偏移，等价于“按类别分组做 NMS”
    max_coord = bboxes.max()
    offsets = labels.to(bboxes) * (max_coord + 1)
    boxes_for_nms = bboxes + offsets[:, None]

    # 先试 mmcv，再退回 torchvision
    out = _nms_mmcv(boxes_for_nms, scores, iou_thr)
    if out is None:
        boxes_keep, scores_keep, keep = _nms_torchvision(boxes_for_nms, scores, iou_thr)
        # 还原到原始坐标
        bboxes_keep = bboxes[keep]
        return bboxes_keep, scores_keep, keep
    else:
        boxes_keep, scores_keep, keep = out
        # 用原始 bboxes 回取
        bboxes_keep = bboxes[keep]
        return bboxes_keep, scores_keep, keep


# ---------------------- multiclass_nms ----------------------
def multiclass_nms(bboxes,
                   scores,
                   score_thr: float = 0.05,
                   nms_cfg: dict = None,
                   max_num: int = 100):
    """
    A robust multi-class NMS that accepts:
      - bboxes: (N, 4) in xyxy
      - scores: (N, C) without background class

    It will internally:
      1) threshold scores > score_thr  -> mask (N*C,)
      2) expand `bboxes` to (N*C, 4) for indexing with that mask
      3) build per-class labels
      4) do class-aware batched NMS
    Return:
      dets: (K,5) [x1,y1,x2,y2,score], labels: (K,)
    """
    assert bboxes.dim() == 2 and bboxes.size(1) == 4, 'bboxes must be (N,4)'
    assert scores.dim() == 2, 'scores should be (N, C)'

    N, C = scores.shape
    if nms_cfg is None:
        nms_cfg = dict(type='nms', iou_threshold=0.5)

    # 1) 有效掩码 (N*C,)
    scores_flat = scores.reshape(-1)
    valid_mask = scores_flat > float(score_thr)
    num_valid = int(valid_mask.sum().item())
    if num_valid == 0:
        empty = bboxes.new_zeros((0, 5))
        labels = bboxes.new_zeros((0,), dtype=torch.long)
        return empty, labels

    # 2) (idx -> (box_idx, cls_idx))
    inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)  # (num_valid,)
    box_inds = (inds // C).to(torch.long)  # (num_valid,)
    cls_inds = (inds % C).to(torch.long)   # (num_valid,)

    # 3) 有效子集
    scores_keep = scores_flat[valid_mask]          # (num_valid,)
    bboxes_keep = bboxes[box_inds]                 # (num_valid, 4)
    labels_keep = cls_inds                         # (num_valid,)

    # 4) 分类感知 NMS
    bboxes_nms, scores_nms, keep = batched_nms(
        bboxes_keep, scores_keep, labels_keep, nms_cfg
    )

    # 5) 组 dets & 截断
    dets = torch.cat([bboxes_nms, scores_nms[:, None]], dim=1)  # (K,5)
    if dets.size(0) > max_num:
        topk = torch.topk(dets[:, -1], k=max_num, dim=0).indices
        dets = dets[topk]
        labels_keep = labels_keep[keep][topk]
    else:
        labels_keep = labels_keep[keep]

    return dets, labels_keep.to(torch.long)
