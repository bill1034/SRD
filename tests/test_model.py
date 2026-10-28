import sys
from pathlib import Path
import torch
import copy
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'cache' / 'dependencies'))
sys.path.insert(0, str(ROOT / 'source'))


@pytest.fixture(scope='module')
def integrated_model():
    from model import build_resroit
    from config import default_config
    cfg = default_config()
    cfg['pretrained_backbone'] = ''
    cfg['score_threshold'], cfg['max_detections'] = .17, 37
    cfg['model']['train_cfg']['rpn']['sampler']['num'] = 64
    for stage in cfg['model']['train_cfg']['rcnn']:
        stage['sampler']['num'] = 32
    for stage in [cfg['model']['train_cfg']['rpn_proposal'], cfg['model']['test_cfg']['rpn']]:
        stage['nms_pre'], stage['max_per_img'] = 100, 64
    cfg['reasoning']['spectral'].update(risk_threshold=0., max_fraction=1., max_rois=64)
    cfg['reasoning']['memory'].update(null_cost=20., capacity_enabled=False)
    cfg['reasoning']['mrf'].update(feature_temperature=100.)
    cfg['ignore_iof_threshold'] = .5
    net = build_resroit(cfg, ['cell'])
    return net, cfg


def test_public_detection_controls_reach_official_head(integrated_model):
    net, cfg = integrated_model
    assert net.test_cfg.rcnn.score_thr == cfg['score_threshold']
    assert net.roi_head.test_cfg.max_per_img == cfg['max_detections']


@pytest.mark.parametrize('with_regular_gt', [True, False])
def test_difficult_candidates_never_become_negatives(integrated_model, with_regular_gt):
    net, _ = integrated_model
    for assigner, rotated in [(net.rpn_head.assigner, False),
                              (net.roi_head.bbox_assigner[0], False),
                              (net.roi_head.bbox_assigner[1], True)]:
        candidates = torch.tensor([[50., 50., 20., 10., .3]]) if rotated else torch.tensor([[40., 40., 60., 60.]])
        regular = torch.tensor([[15., 15., 10., 5., 0.]]) if rotated else torch.tensor([[10., 10., 20., 20.]])
        if not with_regular_gt:
            regular = regular[:0]
        result = assigner.assign(candidates, regular, candidates.clone(), torch.zeros(len(regular), dtype=torch.long))
        assert result.gt_inds.tolist() == [-1]


@pytest.mark.parametrize('field', ['key_dim', 'evidence_dim'])
def test_projection_dimensions_rejected_before_qr(field):
    from config import default_config
    from model import RESRoIHead
    cfg = default_config()
    cfg['reasoning'][field] = cfg['reasoning']['channels'] + 1
    head = copy.deepcopy(cfg['model']['roi_head'])
    head.pop('type')
    with pytest.raises(ValueError, match='dimension|channels'):
        RESRoIHead(cfg['reasoning'], **head)


def test_stage_two_must_use_supported_two_fc_head():
    from config import default_config
    from model import RESRoIHead
    cfg = default_config()
    head = copy.deepcopy(cfg['model']['roi_head'])
    head.pop('type')
    head['bbox_head'][1]['with_avg_pool'] = True
    with pytest.raises(ValueError, match='2FC|two.*FC|average'):
        RESRoIHead(cfg['reasoning'], **head)


def test_quality_targets_are_aligned_iou_for_positives_and_zero_for_background():
    from types import SimpleNamespace
    from model import quality_targets
    positive = torch.tensor([[20., 20., 10., 4., .2],
                             [40., 40., 8., 6., -.3]])
    matched = torch.tensor([[20., 20., 10., 4., .2],
                            [80., 80., 8., 6., -.3]])
    sampled = SimpleNamespace(
        pos_bboxes=positive,
        pos_gt_bboxes=matched,
        neg_bboxes=torch.tensor([[10., 10., 5., 5., 0.]]),
    )
    target = quality_targets([sampled])
    assert target.shape == (3,)
    assert target[0].item() == pytest.approx(1.0, abs=1e-6)
    assert target[1].item() == pytest.approx(0.0, abs=1e-6)
    assert target[2].item() == pytest.approx(0.0, abs=1e-6)


def test_quality_rescore_modulates_foreground_logits_only():
    from model import apply_quality_logit_rescore
    scores = torch.tensor([[2., 1., -1.], [0., 3., 2.]])
    quality_logits = torch.tensor([[0.], [torch.logit(torch.tensor(.8)).item()]])
    rescored = apply_quality_logit_rescore(scores, quality_logits, power=1.0)
    expected_shift = torch.log(torch.tensor([.5, .8]))
    assert torch.allclose(rescored[:, :-1], scores[:, :-1] + expected_shift[:, None])
    assert torch.equal(rescored[:, -1], scores[:, -1])
    assert torch.equal(scores, torch.tensor([[2., 1., -1.], [0., 3., 2.]]))


def test_quality_upgrade_allows_only_new_head_parameters():
    from model import validate_quality_upgrade_keys
    validate_quality_upgrade_keys([
        'roi_head.quality_head.0.weight',
        'roi_head.quality_head.0.bias',
        'roi_head.quality_head.2.weight',
        'roi_head.quality_head.2.bias',
    ], [])
    with pytest.raises(RuntimeError, match='architecture mismatch'):
        validate_quality_upgrade_keys(['roi_head.evidence_down.weight'], [])
    with pytest.raises(RuntimeError, match='architecture mismatch'):
        validate_quality_upgrade_keys([], ['roi_head.unknown.weight'])


def test_quality_loss_is_zero_for_an_empty_sample():
    from model import quality_bce_loss
    logits = torch.empty((0,), requires_grad=True)
    loss = quality_bce_loss(logits, torch.empty((0,)))
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None and logits.grad.numel() == 0

def test_model_source_exists():
    assert (ROOT / 'source' / 'model.py').exists(), 'full integrated model not implemented'

def test_gpu_multiclass_nms_keeps_separate_classes():
    from model import safe_multiclass_nms_rotated
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b = torch.tensor([[20,20,10,4,.3],[20,20,10,4,.3]],device=device,dtype=torch.float32)
    s = torch.tensor([[.9,.8,.01],[.7,.6,.01]],device=device)
    det, labels, inds = safe_multiclass_nms_rotated(b,s,.05,{'iou_thr':.1},20,return_inds=True)
    assert labels.tolist()==[0,1]
    assert inds.tolist()==[0,1]
    assert det.device==b.device

def test_frozen_stage_b_survives_train_call(integrated_model):
    from model import configure_stage
    net, cfg = integrated_model
    configure_stage(net,'B',cfg)
    net.train()
    assert not net.backbone.training
    assert not net.neck.training
    assert not net.rpn_head.training
    assert not net.roi_head.bbox_head[0].training
    assert all(not p.requires_grad for p in net.backbone.parameters())
    assert any(p.requires_grad for p in net.roi_head.bbox_head[1].parameters())


def _summed_loss(losses):
    return sum(sum(value) if isinstance(value, list) else value
               for name, value in losses.items() if 'loss' in name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='This probe verifies actual rotated CUDA operators')
def test_gpu_ignored_geometry_training_and_stage_b_gradients(integrated_model, monkeypatch):
    from model import configure_stage, group_code
    from mmrotate.core import obb2xyxy, rbbox2roi
    net, cfg = integrated_model
    torch.manual_seed(7)
    net.cuda()
    configure_stage(net, 'A', cfg)
    net.train()
    net.zero_grad(set_to_none=True)
    image = torch.randn(1, 3, 128, 128, device='cuda')
    metadata = [dict(img_shape=(128, 128, 3), ori_shape=(128, 128, 3),
                     pad_shape=(128, 128, 3), scale_factor=np.ones(4, dtype=np.float32),
                     flip=False, group_id='query-patient')]
    gt = [torch.tensor([[45., 48., 24., 12., .3], [78., 79., 18., 10., -.4]], device='cuda')]
    labels = [torch.zeros(2, dtype=torch.long, device='cuda')]
    ignored = [torch.tensor([[95., 30., 18., 8., .5]], device='cuda')]
    observed = {}
    for name, assigner in [('rpn', net.rpn_head.assigner),
                           ('s0', net.roi_head.bbox_assigner[0]),
                           ('s1', net.roi_head.bbox_assigner[1])]:
        original = assigner.assign
        def recording(bboxes, normal, ignored_boxes=None, gt_labels=None, _name=name, _original=original):
            observed[_name] = None if ignored_boxes is None else ignored_boxes.detach().clone()
            return _original(bboxes, normal, ignored_boxes, gt_labels)
        monkeypatch.setattr(assigner, 'assign', recording)
    total = _summed_loss(net.forward_train(image, metadata, gt, labels, gt_bboxes_ignore=ignored))
    assert torch.isfinite(total)
    total.backward()
    assert net.backbone.conv1.weights.grad.abs().sum() > 0
    horizontal = obb2xyxy(ignored[0], 'le90')
    assert torch.equal(observed['rpn'], horizontal)
    assert torch.equal(observed['s0'], horizontal)
    assert torch.equal(observed['s1'], ignored[0])
    assert net.roi_head.current_groups == ['query-patient']

    # An image containing only difficult regions must not train them as background.
    net.zero_grad(set_to_none=True)
    only_ignored = [torch.tensor([[64., 64., 512., 512., 0.]], device='cuda')]
    only_loss = _summed_loss(net.forward_train(
        image, metadata, [gt[0][:0]], [labels[0][:0]], gt_bboxes_ignore=only_ignored))
    assert torch.isfinite(only_loss) and only_loss.detach().abs() < 1e-7
    only_loss.backward()

    configure_stage(net, 'B', cfg)
    net.zero_grad(set_to_none=True)
    with torch.no_grad():
        features = net.extract_feat(image)
        extractor = net.roi_head.bbox_roi_extractor[1]
        raw = extractor(features[:extractor.num_inputs], rbbox2roi(gt))
        keys, values = net.roi_head.raw_key_value(raw)
    memory = net.roi_head.memory
    memory.cfg['null_cost'] = 1e6  # Force available evidence for this gradient probe only.
    memory.set_bank(keys.repeat(3, 1), values.repeat(3, 1), labels[0].repeat(3),
                    torch.tensor([group_code(f'reference-{i}') for i in range(3) for _ in range(2)], device='cuda'))
    fixed_bank = memory.bank_keys.clone()
    total_b = _summed_loss(net.forward_train(image, metadata, gt, labels, gt_bboxes_ignore=ignored))
    assert torch.isfinite(total_b)
    total_b.backward()
    for name in ['spectral', 'evidence_down', 'evidence_up', 'memory_out']:
        parameters = list(getattr(net.roi_head, name).parameters())
        assert parameters and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), name
        assert sum(float(p.grad.abs().sum()) for p in parameters) > 0, name
    assert net.roi_head.evidence_scale.grad.abs().sum() > 0
    assert all(p.grad is None for p in net.backbone.parameters())
    assert torch.equal(memory.bank_keys, fixed_bank)
    assert not memory.bank_keys.requires_grad
    diag = net.roi_head.last_diagnostics
    assert diag['spectral']['fft_rois'] > 0
    assert diag['mrf']['edge_count'] > 0
    assert diag['memory']['null_mass'].max() < 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Regression requires a live CUDA model')
def test_e2cnn_basis_cache_isolated_between_live_gpu_and_new_cpu_models(integrated_model):
    from model import build_resroit
    old_model, cfg = integrated_model
    old_model.cuda()
    old_basis = old_model.backbone.conv1._basisexpansion
    gpu_buffers = {name: (buffer, buffer.data_ptr()) for name, buffer in old_basis.named_buffers()}
    assert gpu_buffers and all(buffer.device.type == 'cuda' for buffer, _ in gpu_buffers.values())
    replacement_cfg = copy.deepcopy(cfg)
    replacement_cfg['pretrained_backbone'] = str(ROOT / 'cache/weights/re_resnet50_c8_batch256-25b16846.pth')
    replacement_cfg['pretrained_frozen_stages'] = 1
    # Frozen-stage initialization expands filters before any caller can .cuda().
    # It must use CPU-owned bases even while the previous model is still alive.
    new_model = build_resroit(replacement_cfg, ['cell'], initialize=False)
    assert all(parameter.device.type == 'cpu' for parameter in new_model.parameters())
    assert all(buffer.device.type == 'cpu' for buffer in new_model.buffers())
    assert new_model.backbone.frozen_stages == 1
    assert new_model.backbone.conv1._basisexpansion is not old_basis
    for name, buffer in old_basis.named_buffers():
        previous, pointer = gpu_buffers[name]
        assert buffer is previous and buffer.data_ptr() == pointer and buffer.device.type == 'cuda'
