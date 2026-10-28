"""SRD extensions over the OpenMMLab ReDet implementation.

OpenMMLab components are Apache-2.0, Copyright (c) OpenMMLab.
No user baseline checkpoints or original FAA implementation are bundled.
"""
import copy
import hashlib
import logging
import math
import numpy as np
import torch
from e2cnn.nn.modules.r2_conv import basisexpansion_singleblock as e2cnn_basis_cache
from torch import nn
from torch.nn import functional as F
from mmcv import ConfigDict
from mmcv.ops import nms_rotated
from mmdet.core.bbox.assigners import MaxIoUAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmrotate.core import obb2xyxy, rbbox_overlaps
from mmrotate.models import build_detector
from mmrotate.models.builder import ROTATED_HEADS, ROTATED_DETECTORS
from mmrotate.models.roi_heads.roi_trans_roi_head import RoITransRoIHead
from mmrotate.models.detectors.redet import ReDet
import mmrotate.models.roi_heads.bbox_heads.rotated_bbox_head as rotated_bbox_head_module
from reasoning import SpectralRoIAdapter, BayesianMemory, SmoothTVReasoner


def quality_targets(sampling_results):
    """Return stage-two localization-quality targets in sampled RoI order."""
    targets = []
    for sampled in sampling_results:
        if sampled.pos_bboxes.numel():
            positive = rbbox_overlaps(
                sampled.pos_bboxes.float(),
                sampled.pos_gt_bboxes.float(),
                is_aligned=True,
            ).clamp_(0.0, 1.0)
        else:
            positive = sampled.pos_bboxes.new_zeros((0,), dtype=torch.float32)
        negative = sampled.neg_bboxes.new_zeros((len(sampled.neg_bboxes),), dtype=torch.float32)
        targets.append(torch.cat((positive, negative)))
    if not targets:
        return torch.empty(0)
    return torch.cat(targets)


def apply_quality_logit_rescore(cls_score, quality_logit, power=1.0):
    """Multiply foreground odds by predicted quality while preserving background."""
    if cls_score.ndim != 2 or quality_logit.shape != (cls_score.shape[0], 1):
        raise ValueError('Quality logits must contain one value per RoI')
    if not isinstance(power, (int, float)) or isinstance(power, bool) or not math.isfinite(power) or power < 0:
        raise ValueError('Quality score power must be a finite nonnegative number')
    rescored = cls_score.clone()
    log_quality = F.logsigmoid(quality_logit.float()).to(cls_score.dtype)
    rescored[:, :-1] = rescored[:, :-1] + float(power) * log_quality
    return rescored


def quality_bce_loss(logits, targets):
    if logits.numel() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')


def validate_quality_upgrade_keys(missing_keys, unexpected_keys):
    """Reject checkpoint drift beyond the newly introduced quality head."""
    allowed_prefix = 'roi_head.quality_head.'
    invalid_missing = [key for key in missing_keys if not key.startswith(allowed_prefix)]
    if invalid_missing or unexpected_keys:
        raise RuntimeError(
            f'Checkpoint architecture mismatch: missing={invalid_missing}, extra={list(unexpected_keys)}')


def safe_multiclass_nms_rotated(multi_bboxes, multi_scores, score_thr, nms,
                                max_num=-1, score_factors=None, return_inds=False):
    """Device-safe classwise rotated NMS; indices refer to flattened candidates."""
    nc = multi_scores.shape[1] - 1
    boxes = multi_bboxes.reshape(-1, nc, 5) if multi_bboxes.shape[1] > 5 else multi_bboxes[:, None].expand(-1, nc, -1)
    scores = multi_scores[:, :nc]
    threshold = nms.get('iou_thr', nms.get('iou_threshold', .1))
    kept_boxes, kept_scores, kept_labels, kept_indices = [], [], [], []
    for c in range(nc):
        valid = (scores[:, c] > score_thr) & torch.isfinite(boxes[:, c]).all(-1)
        valid = valid & (boxes[:, c, 2:4] > 0).all(-1)
        ids = torch.where(valid)[0]
        if ids.numel() == 0:
            continue
        bc = boxes[ids, c].float().contiguous()
        sc = scores[ids, c].float()
        if score_factors is not None:
            sc = sc * score_factors[ids]
        _, keep = nms_rotated(bc, sc.contiguous(), float(threshold))
        kept_boxes.append(bc[keep])
        kept_scores.append(sc[keep])
        kept_labels.append(torch.full_like(keep, c))
        kept_indices.append(ids[keep] * nc + c)
    if not kept_boxes:
        det = multi_bboxes.new_zeros((0, 6))
        labels = torch.empty(0, dtype=torch.long, device=multi_scores.device)
        return (det, labels, labels) if return_inds else (det, labels)
    b, s = torch.cat(kept_boxes), torch.cat(kept_scores)
    labels, ids = torch.cat(kept_labels), torch.cat(kept_indices)
    order = torch.argsort(s, descending=True, stable=True)
    if max_num > 0:
        order = order[:max_num]
    det = torch.cat((b[order], s[order, None]), dim=1)
    return (det, labels[order], ids[order]) if return_inds else (det, labels[order])


def group_code(group):
    return int.from_bytes(hashlib.sha256(str(group).encode('utf-8')).digest()[:8], 'big') & ((1 << 63) - 1)


@BBOX_ASSIGNERS.register_module(force=True)
class RESMaxIoUAssigner(MaxIoUAssigner):
    """Official assignment, including ignored regions when regular GT is empty.

    MMDetection 2.28's empty-GT branch otherwise resets every candidate to
    background after computing the ignore overlaps. Nonempty-GT matching and
    sampling remain the official implementation.
    """
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        result = super().assign(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels)
        if (gt_bboxes.numel() == 0 and self.ignore_iof_thr > 0
                and gt_bboxes_ignore is not None and gt_bboxes_ignore.numel()
                and bboxes.numel()):
            if self.ignore_wrt_candidates:
                overlap = self.iou_calculator(bboxes, gt_bboxes_ignore, mode='iof').max(dim=1).values
            else:
                overlap = self.iou_calculator(gt_bboxes_ignore, bboxes, mode='iof').max(dim=0).values
            ignored = overlap > self.ignore_iof_thr
            result.gt_inds[ignored] = -1
            result.max_overlaps[ignored] = -1
            if result.labels is not None:
                result.labels[ignored] = -1
        return result


def ignored_obbs_to_hbbs(boxes, version):
    if boxes is None:
        return None
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError('Ignored boxes must use [cx, cy, w, h, angle] OBB coordinates')
    return obb2xyxy(boxes, version)


@ROTATED_HEADS.register_module(force=True)
class RESRoIHead(RoITransRoIHead):
    def __init__(self, res_cfg, **kwargs):
        self.res_cfg = copy.deepcopy(res_cfg)
        cfg = self.res_cfg
        c, d, k = cfg['channels'], cfg['evidence_dim'], cfg['key_dim']
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (c, d, k)) or not (0 < d <= c and 0 < k <= c):
            raise ValueError('QR key/evidence dimensions must be positive integers no larger than channels')
        super().__init__(**kwargs)
        if self.num_stages != 2:
            raise ValueError('The extension requires exactly two RoI stages and the fixed 2FC head')
        head = self.bbox_head[1]
        if (head.num_shared_convs != 0 or head.num_shared_fcs != 2
                or any(getattr(head, name) != 0 for name in ('num_cls_convs', 'num_cls_fcs', 'num_reg_convs', 'num_reg_fcs'))
                or head.with_avg_pool or not head.with_cls or not head.with_reg):
            raise ValueError('Stage two must use the fixed shared 2FC head without average pooling or separate branches')
        if head.in_channels != c or self.bbox_roi_extractor[1].out_channels != c:
            raise ValueError('Reasoning channels must match the official stage-two RoI extractor and head')
        self.spectral = SpectralRoIAdapter(c, cfg['spectral'])
        self.memory = BayesianMemory(k, d, cfg['memory'])
        self.reasoner = SmoothTVReasoner(cfg['mrf'])
        self.evidence_down = nn.Linear(self.bbox_head[1].fc_out_channels, d)
        self.evidence_up = nn.Linear(d, self.bbox_head[1].fc_out_channels, bias=False)
        self.memory_out = nn.Linear(d, d, bias=False)
        self.evidence_scale = nn.Parameter(torch.full((self.bbox_head[1].fc_out_channels,), cfg['evidence_layer_scale']))
        quality_cfg = copy.deepcopy(cfg.get('quality', {'enabled': False}))
        self.quality_enabled = bool(quality_cfg.get('enabled', False))
        self.quality_loss_weight = float(quality_cfg.get('loss_weight', 0.25))
        self.quality_score_power = float(quality_cfg.get('score_power', 1.0))
        hidden_dim = quality_cfg.get('hidden_dim', 256)
        if (not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim < 1
                or not math.isfinite(self.quality_loss_weight) or self.quality_loss_weight < 0
                or not math.isfinite(self.quality_score_power) or self.quality_score_power < 0):
            raise ValueError('Quality-head dimensions and weights must be finite and nonnegative')
        self.quality_head = (nn.Sequential(
            nn.Linear(self.bbox_head[1].fc_out_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        ) if self.quality_enabled else None)
        generator = torch.Generator().manual_seed(cfg['projection_seed'])
        basis, _ = torch.linalg.qr(torch.randn(c, max(d, k), generator=generator), mode='reduced')
        self.register_buffer('key_projection', basis[:, :k].clone())
        self.register_buffer('value_projection', basis[:, :d].clone())
        self.enhancements_enabled = False
        self.current_groups = None
        self.last_diagnostics = {}

    def raw_key_value(self, roi_features):
        pooled = roi_features.float().mean(dim=(-2, -1))
        return pooled @ self.key_projection, pooled @ self.value_projection

    def shared_features(self, feats):
        head = self.bbox_head[1]
        x = feats.flatten(1)
        for fc in head.shared_fcs:
            x = head.relu(fc(x))
        return x

    def _bbox_forward(self, stage, x, rois):
        if stage == 0 or not self.enhancements_enabled:
            return super()._bbox_forward(stage, x, rois)
        extractor = self.bbox_roi_extractor[1]
        raw = extractor(x[:extractor.num_inputs], rois)
        head = self.bbox_head[1]
        if rois.shape[0] == 0:
            cls, reg = head(raw)
            self.last_diagnostics = {'action': 'empty_proposals', 'rois': 0}
            result = dict(cls_score=cls, bbox_pred=reg, bbox_feats=raw)
            if self.quality_enabled:
                result['quality_logit'] = cls.new_zeros((0, 1))
            return result
        raw_keys, _ = self.raw_key_value(raw.detach())
        h_shared = self.shared_features(raw)
        class_prob = head.fc_cls(h_shared).float().softmax(-1).detach()
        if self.res_cfg['spectral_enabled']:
            adapted, spectral_diag = self.spectral(raw, rois.detach(), class_prob=class_prob)
            selected = spectral_diag['selected_indices']
            if selected.numel():
                h_shared = h_shared.clone()
                h_shared[selected] = self.shared_features(adapted[selected])
        else:
            spectral_diag = {'fft_rois': 0, 'action': 'disabled'}
        with torch.autocast(device_type=raw.device.type, enabled=False):
            h0 = self.evidence_down(h_shared.float())
            if self.res_cfg['memory_enabled']:
                query_groups = None
                if self.training and self.current_groups is not None:
                    all_groups = torch.tensor([group_code(g) for g in self.current_groups], device=rois.device, dtype=torch.long)
                    query_groups = all_groups[rois[:, 0].long()]
                r, reliability, memory_diag = self.memory(raw_keys.float(), query_groups, query_image_ids=rois[:, 0].long())
                r = self.memory_out(r.float())
            else:
                r, reliability = torch.zeros_like(h0), h0.new_zeros(h0.shape[0])
                memory_diag = {'action': 'disabled'}
            if self.res_cfg['mrf_enabled'] or self.res_cfg['memory_enabled']:
                h, graph_diag = self.reasoner(h0, r, reliability, rois.float(), raw_keys.float())
                h_shared = h_shared.float() + self.evidence_scale * self.evidence_up(h - h0)
            else:
                graph_diag = {'action': 'disabled'}
        cls, reg = head.fc_cls(h_shared), head.fc_reg(h_shared)
        quality_logit = self.quality_head(h_shared) if self.quality_enabled else None
        # RoITrans averages two stages. Doubling the stage-two adjustment makes
        # the final averaged foreground log-odds receive exactly score_power.
        if quality_logit is not None and not self.training:
            cls = apply_quality_logit_rescore(
                cls, quality_logit, power=2.0 * self.quality_score_power)
        self.last_diagnostics = {'action': 'evidence_refinement', 'rois': int(rois.shape[0]),
                                 'spectral': spectral_diag, 'memory': memory_diag, 'mrf': graph_diag}
        result = dict(cls_score=cls, bbox_pred=reg, bbox_feats=raw)
        if quality_logit is not None:
            result['quality_logit'] = quality_logit
        return result

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        result = super()._bbox_forward_train(
            stage, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        if stage == 1 and self.quality_enabled and self.enhancements_enabled:
            target = quality_targets(sampling_results).to(result['quality_logit'])
            if target.shape[0] != result['quality_logit'].shape[0]:
                raise RuntimeError('Quality target count differs from stage-two RoI count')
            logits = result['quality_logit'].squeeze(1)
            loss = quality_bce_loss(logits, target)
            result['loss_bbox']['loss_quality'] = loss * self.quality_loss_weight
            with torch.no_grad():
                result['loss_bbox']['quality_mae'] = ((logits.sigmoid() - target).abs().mean()
                                                        if logits.numel() else logits.sum() * 0.0)
        return result

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None):
        """Official cascade flow with ignore boxes in each stage's coordinates.

        Adapted from MMRotate 0.3.4 RoITransRoIHead.forward_train (Apache-2.0).
        Assignment, sampling, targets, losses and detached refinement are kept.
        """
        self.current_groups = [m['group_id'] for m in img_metas]
        ignored = gt_bboxes_ignore if gt_bboxes_ignore is not None else [None] * len(img_metas)
        if len(ignored) != len(img_metas):
            raise ValueError('Ignored box list must contain one entry per image')
        losses = {}
        for stage in range(self.num_stages):
            self.current_stage = stage
            train_cfg = self.train_cfg[stage]
            sampling_results = []
            for image_index in range(len(img_metas)):
                stage_gt = obb2xyxy(gt_bboxes[image_index], self.version) if stage == 0 else gt_bboxes[image_index]
                stage_ignore = ignored_obbs_to_hbbs(ignored[image_index], self.version) if stage == 0 else ignored[image_index]
                assigned = self.bbox_assigner[stage].assign(
                    proposal_list[image_index], stage_gt, stage_ignore, gt_labels[image_index])
                sampled = self.bbox_sampler[stage].sample(
                    assigned, proposal_list[image_index], stage_gt, gt_labels[image_index],
                    feats=[level[image_index][None] for level in x])
                sampled.pos_gt_bboxes = gt_bboxes[image_index][sampled.pos_assigned_gt_inds]
                sampling_results.append(sampled)
            bbox_results = self._bbox_forward_train(
                stage, x, sampling_results, gt_bboxes, gt_labels, train_cfg)
            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = value * self.stage_loss_weights[stage] if 'loss' in name else value
            if stage < self.num_stages - 1:
                pos_is_gts = [sample.pos_is_gt for sample in sampling_results]
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[stage].custom_activation:
                        cls_score = self.bbox_head[stage].loss_cls.get_activation(cls_score)
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[stage].num_classes,
                        cls_score[:, :-1].argmax(1), roi_labels)
                    proposal_list = self.bbox_head[stage].refine_bboxes(
                        bbox_results['rois'], roi_labels, bbox_results['bbox_pred'], pos_is_gts, img_metas)
        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        self.current_groups = None
        return super().simple_test(x, proposal_list, img_metas, rescale=rescale)


@ROTATED_DETECTORS.register_module(force=True)
class RESReDet(ReDet):
    """Official ReDet with a stage-aware freeze contract."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.res_stage = 'A'

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
        """Official two-stage flow; RPN's horizontal assigner receives HBB ignores."""
        features = self.extract_feat(img)
        losses = {}
        if self.with_rpn:
            ignored_hbbs = None if gt_bboxes_ignore is None else [
                ignored_obbs_to_hbbs(boxes, self.rpn_head.version) for boxes in gt_bboxes_ignore]
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                features, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=ignored_hbbs, proposal_cfg=proposal_cfg, **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        losses.update(self.roi_head.forward_train(
            features, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs))
        return losses

    def train(self, mode=True):
        super().train(mode)
        if self.res_stage == 'B':
            self.backbone.eval()
            self.neck.eval()
            self.rpn_head.eval()
            self.roi_head.bbox_head[0].eval()
        return self


def configure_stage(model, stage, cfg):
    if stage not in ('A', 'B'):
        raise ValueError('stage must be A or B')
    model.res_stage = stage
    for p in model.parameters():
        p.requires_grad_(True)
    model.roi_head.enhancements_enabled = stage == 'B'
    if stage == 'A':
        for module in [model.roi_head.spectral, model.roi_head.memory, model.roi_head.reasoner,
                       model.roi_head.evidence_down, model.roi_head.evidence_up, model.roi_head.memory_out]:
            for p in module.parameters():
                p.requires_grad_(False)
        model.roi_head.evidence_scale.requires_grad_(False)
        if model.roi_head.quality_head is not None:
            for p in model.roi_head.quality_head.parameters():
                p.requires_grad_(False)
    else:
        for module in [model.backbone, model.neck, model.rpn_head, model.roi_head.bbox_head[0]]:
            for p in module.parameters():
                p.requires_grad_(False)
    model.train(model.training)
    return model


def build_resroit(cfg, classes, initialize=True):
    # e2cnn 0.2.3 caches mutable nn.Modules, whose sampled_basis buffers follow
    # an existing model onto CUDA. A new CPU model must not share those modules.
    # Releasing cache references preserves every live model's owned bases.
    e2cnn_basis_cache._stored_filters.clear()
    rotated_bbox_head_module.multiclass_nms_rotated = safe_multiclass_nms_rotated
    for name in ('mmcv', 'mmrotate', 'mmdet'):
        logging.getLogger(name).setLevel(logging.ERROR)
    mc = copy.deepcopy(cfg['model'])
    mc['type'] = 'RESReDet'
    mc['roi_head']['type'] = 'RESRoIHead'
    mechanism = copy.deepcopy(cfg['reasoning'])
    if not mechanism['mrf_enabled']:
        mechanism['mrf']['lambda_pairwise'] = 0.0
    mechanism['memory']['num_classes'] = len(classes)
    mc['roi_head']['res_cfg'] = mechanism
    ignore_threshold = float(cfg.get('ignore_iof_threshold', .5))
    if not 0 < ignore_threshold <= 1:
        raise ValueError('ignore_iof_threshold must lie in (0, 1]')
    for stage in [mc['train_cfg']['rpn'], *mc['train_cfg']['rcnn']]:
        if stage['assigner']['type'] not in ('MaxIoUAssigner', 'RESMaxIoUAssigner'):
            raise ValueError('The fixed ReDet cascade requires the official MaxIoU assignment contract')
        stage['assigner'].update(type='RESMaxIoUAssigner', ignore_iof_thr=ignore_threshold)
    score_threshold, max_detections = float(cfg['score_threshold']), int(cfg['max_detections'])
    if not 0 <= score_threshold <= 1 or max_detections < 1:
        raise ValueError('score_threshold must lie in [0,1] and max_detections must be positive')
    mc['test_cfg']['rcnn'].update(score_thr=score_threshold, max_per_img=max_detections)
    for h in mc['roi_head']['bbox_head']:
        h['num_classes'] = len(classes)
    pretrained = cfg['pretrained_backbone']
    mc['backbone']['pretrained'] = pretrained or None
    mc['backbone']['frozen_stages'] = cfg['pretrained_frozen_stages'] if pretrained else -1
    model = build_detector(ConfigDict(mc))
    if initialize:
        model.init_weights()
    model.CLASSES = tuple(classes)
    return configure_stage(model, 'A', cfg)
