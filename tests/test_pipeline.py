import sys
from pathlib import Path
import pytest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'cache' / 'dependencies'))
sys.path.insert(0, str(ROOT / 'source'))

def test_signature_catches_labels_but_allows_relocation():
    from pipeline import dataset_signature
    a = dict(id='a', image='C:/one/a.png', split='train', group_id='p1', pixel_sha256='abc',
             boxes=np.array([[20,20,10,5,.2]]), labels=np.array([0]), ignored_boxes=np.empty((0,5)), ignored_labels=np.array([],dtype=int))
    b = dict(a, image='D:/new/a.png')
    assert dataset_signature([a], ['cell']) == dataset_signature([b], ['cell'])
    b = dict(a, labels=np.array([1]))
    assert dataset_signature([a], ['cell']) != dataset_signature([b], ['cell'])

def test_rng_roundtrip():
    import random
    from pipeline import capture_rng, restore_rng
    state = capture_rng()
    wanted = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng(state)
    assert random.random() == wanted[0]
    assert np.random.rand() == wanted[1]
    assert torch.equal(torch.rand(1), wanted[2])

def test_runtime_rejects_nonworking_controls():
    from pipeline import validate_config
    from config import default_config
    cfg = default_config()
    cfg['accumulation_steps'] = 0
    with pytest.raises(ValueError, match='accumulation_steps'):
        validate_config(cfg)

def test_checkpoint_pair_rejects_another_experiment(tmp_path):
    from pipeline import verify_checkpoint_pair
    a=dict(run_id='experiment-a',dataset_signature='data',classes=['cell'],stage='B',bank_keyspace='space',best_metric=.3,epoch=3)
    b=dict(a,run_id='experiment-b',epoch=5)
    path=tmp_path/'best.pth';torch.save(a,path)
    with pytest.raises(ValueError,match='run_id'):
        verify_checkpoint_pair(path,b)
    verify_checkpoint_pair(path,dict(a,epoch=5))

def test_local_pretrained_hash_mismatch_rejected(tmp_path):
    from pipeline import prepare_pretrained
    path=tmp_path/'bad.pth';path.write_bytes(b'not-an-official-weight')
    with pytest.raises(ValueError,match='SHA256'):
        prepare_pretrained(dict(pretrained_backbone=str(path),pretrained_sha256='0'*64))

def test_checkpoint_filter_caches_are_regenerated(tmp_path):
    from pipeline import save_checkpoint, load_checkpoint, keyspace_signature
    from config import default_config
    from model import build_resroit, configure_stage
    cfg=default_config(); cfg['pretrained_backbone']=''
    net=build_resroit(cfg,['cell']); configure_stage(net,'B',cfg)
    head=net.roi_head
    head.memory.set_bank(torch.randn(3,32),torch.randn(3,192),torch.zeros(3,dtype=torch.long),torch.arange(3))
    net.eval()
    p=tmp_path/'roundtrip.pth'
    save_checkpoint(p,net,cfg,['cell'],'B',1,0.2,0,'sig',None,None,[], keyspace_signature(net))
    loaded,payload=load_checkpoint(p,'cpu')
    assert loaded.res_stage=='B'
    assert torch.equal(head.memory.bank_keys,loaded.roi_head.memory.bank_keys)
    key=next(iter(net.backbone.state_dict()))
    assert torch.equal(net.backbone.state_dict()[key],loaded.backbone.state_dict()[key])
    assert payload['dataset_signature']=='sig'
