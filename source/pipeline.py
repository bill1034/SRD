"""Training, frozen train-only memory, safe checkpoints and image-only inference."""
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
import urllib.request
import uuid
from pathlib import Path
import numpy as np
import torch
import mmcv
import mmdet
import mmrotate
from torch.utils.data import DataLoader
from model import build_resroit, configure_stage, group_code
from data import load_data, OBBDataset, collate_obbs, inference_records, evaluate_obbs, save_predictions


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [plain(v) for v in value]
    return value


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def dataset_signature(records, classes):
    keys=('id','split','group_id','pixel_sha256','boxes','labels','ignored_boxes','ignored_labels')
    payload=dict(classes=classes,records=[{k:plain(r.get(k)) for k in keys} for r in sorted(records,key=lambda r:r['id'])])
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def capture_rng():
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy=[state[0],state[1].tolist(),int(state[2]),int(state[3]),float(state[4])],
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy']; np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the same CUDA device count for RNG replay')
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def validate_config(cfg):
    if cfg['mode'] not in ('train','resume','infer','evaluate'):
        raise ValueError('mode must be train / resume / infer / evaluate')
    for key in ('image_size','batch_size','accumulation_steps','num_threads','stage_a_epochs','stage_b_epochs','validation_interval','checkpoint_every','bank_max_instances_per_chunk'):
        if not isinstance(cfg[key],int) or isinstance(cfg[key],bool) or cfg[key]<=0:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['image_size']%32:
        raise ValueError('image_size must be divisible by 32')
    if cfg['num_workers'] != 0:
        raise ValueError('This portable notebook uses num_workers=0; notebook-defined Dataset classes cannot be spawned safely on Windows')
    if cfg['optimizer'] not in ('SGD','AdamW'):
        raise ValueError('optimizer must be SGD or AdamW')
    for key in ('learning_rate_a','learning_rate_b','gradient_clip'):
        if not math.isfinite(cfg[key]) or cfg[key]<=0:
            raise ValueError(f'{key} must be finite and positive')
    if not 0<=cfg['score_threshold']<=1 or cfg['max_detections']<=0:
        raise ValueError('invalid score_threshold or max_detections')
    if cfg['amp']:
        raise ValueError('amp=True is not supported by the pinned C8/RiRoI engine. Keep amp=False for verified FP32 execution.')
    if cfg['device']=='auto':
        cfg['device']='cuda' if torch.cuda.is_available() else 'cpu'
    if str(cfg['device']).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if not str(mmcv.__version__).startswith('1.7.') or mmdet.__version__!='2.28.2' or mmrotate.__version__!='0.3.4':
        raise RuntimeError('Use mmcv-full 1.7.2, mmdet 2.28.2 and mmrotate 0.3.4; newer APIs differ')
    if not str(cfg['device']).startswith('cuda'):
        raise RuntimeError('Full RiRoIAlignRotated detection requires the verified CUDA environment. CPU is supported only for isolated math/data tests.')
    return cfg


def setup_seed(cfg):
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed']); torch.set_num_threads(cfg['num_threads'])
    torch.backends.cudnn.benchmark=not cfg['deterministic']
    torch.backends.cudnn.deterministic=cfg['deterministic']


def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_pretrained(cfg):
    if cfg['pretrained_backbone']:
        path=Path(cfg['pretrained_backbone']).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'pretrained_backbone not found: {path}')
        if file_sha256(path)!=cfg['pretrained_sha256']:
            raise ValueError('Local pretrained SHA256 differs from the specified official checkpoint')
        cfg['pretrained_backbone']=str(path)
        return
    if not cfg['download_pretrained']:
        return
    path=Path(cfg['output_dir'])/'cache'/'re_resnet50_c8_batch256-25b16846.pth'
    path.parent.mkdir(parents=True,exist_ok=True)
    expected=cfg['pretrained_sha256']
    if not path.exists() or file_sha256(path)!=expected:
        temporary=path.with_suffix('.download')
        print('Downloading verified official C8 ImageNet initialization...')
        urllib.request.urlretrieve(cfg['pretrained_url'],temporary)
        if file_sha256(temporary)!=expected:
            raise RuntimeError('Downloaded pretrained SHA256 mismatch; file was not loaded')
        os.replace(temporary,path)
    cfg['pretrained_backbone']=str(path.resolve())


def keyspace_signature(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if name.startswith(('backbone.','neck.')) or name in ('roi_head.key_projection','roi_head.value_projection'):
            if name.endswith(('.filter','.expanded_bias')):
                continue
            digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model, cfg, classes, stage, epoch, best_metric, bad_epochs,
                    signature, optimizer, scaler, history, bank_keyspace):
    # e2cnn expanded filters are derived caches, not independent model state.
    weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.endswith(('.filter','.expanded_bias'))}
    payload=dict(format='RES-RoIT-BM-1',run_id=cfg.get('_run_id','unspecified'),weights=weights,cfg=copy.deepcopy(cfg),classes=list(classes),stage=stage,
                 epoch=int(epoch),best_metric=float(best_metric),bad_epochs=int(bad_epochs),dataset_signature=signature,
                 optimizer=optimizer.state_dict() if optimizer else None,scaler=scaler.state_dict() if scaler else None,
                 history=copy.deepcopy(history),rng=capture_rng(),bank_keyspace=bank_keyspace,
                 versions=dict(torch=str(torch.__version__),mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmrotate=mmrotate.__version__),
                 checkpoint_contract='completed epoch; no partial accumulation state')
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(payload,tmp); os.replace(tmp,path)


def load_checkpoint(path, device):
    payload=torch.load(str(path),map_location='cpu',weights_only=True)
    if payload.get('format')!='RES-RoIT-BM-1':
        raise ValueError('Not a compatible SRD checkpoint')
    # initialize=False never reads the original pretrained file. Preserve the
    # frozen-stage setting when resuming an ImageNet-initialized Stage A.
    cfg=copy.deepcopy(payload['cfg'])
    model=build_resroit(cfg,payload['classes'],initialize=False)
    result=model.load_state_dict(payload['weights'],strict=False)
    missing=[k for k in result.missing_keys if not k.endswith(('.filter','.expanded_bias'))]
    if missing or result.unexpected_keys:
        raise RuntimeError(f'Checkpoint architecture mismatch: missing={missing}, extra={result.unexpected_keys}')
    # Go through train before eval to invalidate any stale expanded C8 filters.
    model.train(); model.to(device); configure_stage(model,payload['stage'],payload['cfg']); model.eval()
    if payload['stage']=='B' and payload['bank_keyspace'] and keyspace_signature(model)!=payload['bank_keyspace']:
        raise RuntimeError('Frozen reference-bank key space differs from checkpoint backbone/FPN')
    return model,payload


def compact_diagnostics(value):
    if isinstance(value,torch.Tensor):
        v=value.detach()
        if v.numel()<=12:
            return plain(v)
        return dict(shape=list(v.shape),min=float(v.min()),max=float(v.max()),mean=float(v.float().mean()))
    if isinstance(value,dict):
        return {k:compact_diagnostics(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [compact_diagnostics(v) for v in value]
    return plain(value)


@torch.no_grad()
def predict_records(model, records, cfg):
    model.eval(); device=next(model.parameters()).device
    predictions={}; logs=[]; timings=[]
    dataset=OBBDataset(records,cfg)
    for item,record in zip(dataset,records):
        if device.type=='cuda': torch.cuda.synchronize(device)
        start=time.perf_counter()
        result=model.simple_test(item['img'][None].to(device),[item['meta']],rescale=True)[0]
        if device.type=='cuda': torch.cuda.synchronize(device)
        timings.append(time.perf_counter()-start)
        boxes=[]; scores=[]; labels=[]
        for label,arr in enumerate(result):
            arr=np.asarray(arr,dtype=np.float32).reshape(-1,6)
            boxes.extend(arr[:,:5].tolist()); scores.extend(arr[:,5].tolist()); labels.extend([label]*len(arr))
        predictions[record['id']]=dict(boxes=boxes,scores=scores,labels=labels)
        logs.append(dict(id=record['id'],diagnostics=compact_diagnostics(model.roi_head.last_diagnostics)))
    timing=dict(images=len(timings),seconds=sum(timings),seconds_per_image=float(np.mean(timings)) if timings else None,
                scope='device transfer + model + NMS; excludes disk image decode, metric and overlay; includes first-call warmup')
    return predictions,logs,timing


@torch.no_grad()
def build_reference_bank(model, train_records, cfg):
    if any(r['split']!='train' for r in train_records):
        raise ValueError('Only train annotations may enter the memory bank')
    model.eval(); device=next(model.parameters()).device
    keys=[]; values=[]; labels=[]; groups=[]
    extractor=model.roi_head.bbox_roi_extractor[1]
    for item in OBBDataset(train_records,cfg):
        boxes=item['boxes'].to(device)
        if not len(boxes): continue
        features=model.extract_feat(item['img'][None].to(device))
        for start in range(0,len(boxes),cfg['bank_max_instances_per_chunk']):
            selected=boxes[start:start+cfg['bank_max_instances_per_chunk']]
            rois=torch.cat((selected.new_zeros(len(selected),1),selected),1)
            raw=extractor(features[:extractor.num_inputs],rois)
            k,v=model.roi_head.raw_key_value(raw)
            keys.append(k.cpu()); values.append(v.cpu())
            labels.append(item['labels'][start:start+len(selected)])
            groups.append(torch.full((len(selected),),group_code(item['meta']['group_id']),dtype=torch.long))
    if not keys:
        raise ValueError('Training data has no positive annotated boxes; cannot train this detector or build a bank')
    model.roi_head.memory.set_bank(torch.cat(keys).to(device),torch.cat(values).to(device),torch.cat(labels).to(device),torch.cat(groups).to(device),source_split='train')
    return keyspace_signature(model)


def make_optimizer(model,cfg,stage):
    params=[p for p in model.parameters() if p.requires_grad]
    lr=cfg['learning_rate_a' if stage=='A' else 'learning_rate_b']
    if cfg['optimizer']=='SGD':
        return torch.optim.SGD(params,lr=lr,momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    return torch.optim.AdamW(params,lr=lr,weight_decay=cfg['weight_decay'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'])


def stage_learning_rate(cfg,stage,epoch,update,updates_per_epoch,total_epochs):
    base=cfg['learning_rate_a' if stage=='A' else 'learning_rate_b']
    step=epoch*updates_per_epoch+update
    warm=min(1.,step/max(1,cfg['warmup_steps'])) if cfg['warmup_steps'] else 1.
    warm=cfg['warmup_ratio']+(1-cfg['warmup_ratio'])*warm
    decay=sum(epoch>=max(1,int(total_epochs*m)) for m in cfg['lr_decay_milestones'])
    return base*warm*cfg['lr_decay_gamma']**decay


def verify_checkpoint_pair(best_path,resume):
    selected=torch.load(best_path,map_location='cpu',weights_only=True)
    for key in ('run_id','dataset_signature','classes','stage','bank_keyspace','best_metric'):
        if selected.get(key)!=resume.get(key):
            raise ValueError(f'Best/last checkpoint pair differs in {key}; copy the matching pair from this experiment')
    if selected['epoch']>resume['epoch']:
        raise ValueError('The best checkpoint comes from a later epoch than the resume checkpoint')


def train_stage(model,stage,train_records,val_records,classes,cfg,signature,bank_keyspace,resume=None):
    configure_stage(model,stage,cfg)
    optimizer=make_optimizer(model,cfg,stage)
    epochs=cfg['stage_a_epochs' if stage=='A' else 'stage_b_epochs']
    first=0; best=-1.; bad=0; history=[]
    output=Path(cfg['output_dir']); best_path=output/f'stage_{stage}_best.pth'
    if resume:
        first=resume['epoch']; best=resume['best_metric']; bad=resume['bad_epochs']; history=resume['history']
        optimizer.load_state_dict(resume['optimizer']); restore_rng(resume['rng'])
        if first and not best_path.is_file():
            raise FileNotFoundError(f'Resume needs the corresponding best checkpoint beside last: {best_path}')
        if first:
            verify_checkpoint_pair(best_path,resume)
    loader=DataLoader(OBBDataset(train_records,cfg,training=True),batch_size=cfg['batch_size'],shuffle=True,
                      num_workers=0,collate_fn=collate_obbs,pin_memory=True,drop_last=False)
    device=next(model.parameters()).device
    updates=math.ceil(len(loader)/cfg['accumulation_steps'])
    for epoch in range(first,epochs):
        if bad>=cfg['early_stopping_patience'] and epoch>=cfg['early_stopping_start_epoch']:
            break
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss_totals={}; start=time.perf_counter()
        for index,batch in enumerate(loader):
            window_start=(index//cfg['accumulation_steps'])*cfg['accumulation_steps']
            window_images=min(cfg['accumulation_steps']*cfg['batch_size'],len(train_records)-window_start*cfg['batch_size'])
            batch_weight=len(batch['metas'])/window_images
            lr=stage_learning_rate(cfg,stage,epoch,index//cfg['accumulation_steps'],updates,epochs)
            for group in optimizer.param_groups: group['lr']=lr
            losses=model.forward_train(batch['img'].to(device,non_blocking=True),batch['metas'],
                                       [v.to(device) for v in batch['boxes']],[v.to(device) for v in batch['labels']],
                                       gt_bboxes_ignore=[v.to(device) for v in batch['ignored_boxes']])
            reduced={k:(sum(vv.mean() for vv in v) if isinstance(v,list) else v.mean()) for k,v in losses.items()}
            total=sum(v for k,v in reduced.items() if 'loss' in k)
            if not torch.isfinite(total):
                raise FloatingPointError(f'Nonfinite loss at stage {stage}, epoch {epoch+1}, batch {index}')
            (total*batch_weight).backward()
            for key,value in reduced.items(): loss_totals[key]=loss_totals.get(key,0.)+float(value.detach())
            if (index+1)%cfg['accumulation_steps']==0 or index+1==len(loader):
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        metrics=None; improved=False
        # Always validate the first/final epoch to guarantee a usable selected checkpoint.
        if (epoch+1)%cfg['validation_interval']==0 or epoch==0 or epoch+1==epochs:
            predictions,_,_=predict_records(model,val_records,cfg)
            metrics=evaluate_obbs(predictions,val_records,len(classes),cfg)
            score=metrics.get(cfg['monitor_metric'])
            if score is None or not math.isfinite(score):
                raise ValueError('Validation monitor is undefined; validation must contain positive ground truth and a supported metric')
            if score>best+cfg['early_stopping_min_delta']:
                best=float(score); bad=0; improved=True
            elif epoch+1>=cfg['early_stopping_start_epoch']:
                bad+=1
        row=dict(stage=stage,epoch=epoch+1,losses={k:v/len(loader) for k,v in loss_totals.items()},
                 validation=metrics,learning_rate=lr,seconds=time.perf_counter()-start,
                 gradient_norm=float(norm),diagnostics=compact_diagnostics(model.roi_head.last_diagnostics))
        history.append(row); write_json(output/f'history_{stage}.json',history)
        if improved:
            save_checkpoint(best_path,model,cfg,classes,stage,epoch+1,best,bad,signature,optimizer,None,history,bank_keyspace)
        if (epoch+1)%cfg['checkpoint_every']==0 or epoch+1==epochs or bad>=cfg['early_stopping_patience']:
            save_checkpoint(output/f'stage_{stage}_last.pth',model,cfg,classes,stage,epoch+1,best,bad,signature,optimizer,None,history,bank_keyspace)
        print(f'Stage {stage} epoch {epoch+1}/{epochs}: loss={sum(v for k,v in row["losses"].items() if "loss" in k):.4f}; val {cfg["monitor_metric"]}={None if metrics is None else metrics[cfg["monitor_metric"]]}; best={best:.6f}')
    if not best_path.is_file():
        raise RuntimeError('No validated checkpoint was produced')
    return best_path


def run_model(user_cfg):
    cfg=copy.deepcopy(user_cfg)
    output=Path(cfg['output_dir']).expanduser().resolve(); cfg['output_dir']=str(output)
    output.mkdir(parents=True,exist_ok=True)
    validate_config(cfg); setup_seed(cfg)
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    environment=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                     device=torch.cuda.get_device_name(torch.device(cfg['device'])),mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmrotate=mmrotate.__version__,
                     determinism='seeded; custom CUDA kernels may remain nondeterministic')
    mode=cfg['mode']; checkpoint=cfg['checkpoint_path']
    if mode in ('infer','evaluate'):
        if not checkpoint: raise ValueError('Set checkpoint_path to your trained SRD checkpoint')
        model,payload=load_checkpoint(checkpoint,cfg['device']); classes=payload['classes']
        # Preprocessing/model controls are checkpoint-bound. Explicit overrides are logged.
        runtime=copy.deepcopy(payload['cfg'])
        for k in ('device','output_dir','save_overlays','overlay_score_threshold','overlay_line_width','eval_device','metric_score_threshold','metric_match_iou'):
            runtime[k]=cfg[k]
        for k,v in cfg['inference_overrides'].items():
            if k not in ('score_threshold','max_detections'):
                raise ValueError(f'Unsupported inference override: {k}')
            if k=='score_threshold' and not (isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1):
                raise ValueError('Inference score_threshold must lie in [0,1]')
            if k=='max_detections' and not (isinstance(v,int) and not isinstance(v,bool) and v>0):
                raise ValueError('Inference max_detections must be a positive integer')
            runtime[k]=v
        model.roi_head.test_cfg.score_thr=runtime['score_threshold']; model.roi_head.test_cfg.max_per_img=runtime['max_detections']
        if mode=='infer':
            records=inference_records(cfg['inference_dir'],exclude_dirs=(output,)); audit=None
        else:
            declared,all_records,audit=load_data(cfg,required_splits=(cfg['evaluation_split'],))
            if declared!=classes: raise ValueError('Evaluation class order differs from checkpoint')
            records=[r for r in all_records if r['split']==cfg['evaluation_split']]
            if not records: raise ValueError('Requested evaluation split is empty')
        predictions,logs,timing=predict_records(model,records,runtime)
        prediction_file=save_predictions(predictions,records,classes,output,runtime)
        metrics=evaluate_obbs(predictions,records,len(classes),runtime) if mode=='evaluate' else None
        write_json(output/'agent_log.json',logs)
        result=dict(mode=mode,classes=classes,checkpoint=str(Path(checkpoint).resolve()),prediction_file=prediction_file,
                    metrics=metrics,timing=timing,environment=environment,data_audit=audit,inference_overrides=cfg['inference_overrides'])
    else:
        classes,records,audit=load_data(cfg)
        # The fingerprint is order invariant; canonicalize iteration as well so
        # relocated/reordered manifests retain the same seeded resume trajectory.
        records=sorted(records,key=lambda record:record['id'])
        signature=dataset_signature(records,classes)
        train_records=[r for r in records if r['split']=='train']; val_records=[r for r in records if r['split']=='val']
        if not any(len(r['boxes']) for r in train_records) or not any(len(r['boxes']) for r in val_records):
            raise ValueError('Training and validation splits each require positive annotated objects')
        write_json(output/'data_audit.json',audit)
        resume=None; bank_keyspace=None
        if mode=='resume':
            if not checkpoint: raise ValueError('Resume requires checkpoint_path')
            model,resume=load_checkpoint(checkpoint,cfg['device'])
            if signature!=resume['dataset_signature'] or classes!=resume['classes']:
                raise ValueError('Resume dataset/class fingerprint changed; start a new run instead')
            # Resume preserves the experiment; only paths/runtime rendering may move.
            movable={'mode','manifest_path','data_root','inference_dir','output_dir','checkpoint_path','classes','groups_csv',
                     'download_pretrained','pretrained_url','pretrained_sha256','pretrained_backbone','device','num_threads',
                     'evaluate_test','save_overlays','evaluation_split','inference_overrides','overlay_score_threshold','overlay_line_width','eval_device'}
            for key in set(cfg)-movable:
                if cfg[key]!=resume['cfg'][key]: raise ValueError(f'Resume setting changed: {key}')
            cfg['pretrained_backbone']=resume['cfg']['pretrained_backbone']
            cfg['_run_id']=resume['run_id']
            bank_keyspace=resume['bank_keyspace']
        else:
            if (output/'stage_A_best.pth').exists() or (output/'stage_B_best.pth').exists():
                raise FileExistsError('Training output already contains checkpoints. Use mode=resume or choose a new output_dir.')
            cfg['_run_id']=uuid.uuid4().hex
            prepare_pretrained(cfg); model=build_resroit(cfg,classes).to(cfg['device'])
        write_json(output/'resolved_config.json',cfg)
        if resume is None or resume['stage']=='A':
            a_path=train_stage(model,'A',train_records,val_records,classes,cfg,signature,None,resume)
            del model
            torch.cuda.empty_cache()
            model,_=load_checkpoint(a_path,cfg['device'])
            bank_keyspace=build_reference_bank(model,train_records,cfg)
            write_json(output/'memory_audit.json',dict(source='train-only; best Stage A raw aligned GT RoIs',keyspace=bank_keyspace,
                        entries=int(model.roi_head.memory.bank_keys.shape[0]),query_exclusion='same group excluded during training',
                        inference_updates=False,aggregation='one mean key/value per class and group'))
            resume=None
        b_path=train_stage(model,'B',train_records,val_records,classes,cfg,signature,bank_keyspace,resume)
        del model
        torch.cuda.empty_cache()
        model,b_payload=load_checkpoint(b_path,cfg['device'])
        predictions,logs,timing=predict_records(model,val_records,cfg)
        val_metrics=evaluate_obbs(predictions,val_records,len(classes),cfg)
        save_predictions(predictions,val_records,classes,output/'validation',cfg)
        write_json(output/'validation'/'agent_log.json',logs)
        test_metrics=None
        test_records=[r for r in records if r['split']=='test']
        if cfg['evaluate_test'] and test_records:
            test_predictions,test_logs,test_timing=predict_records(model,test_records,cfg)
            test_metrics=evaluate_obbs(test_predictions,test_records,len(classes),cfg)
            save_predictions(test_predictions,test_records,classes,output/'test',cfg)
            write_json(output/'test'/'agent_log.json',test_logs)
        result=dict(mode=mode,model='SRD',classes=classes,checkpoint=str(b_path),data_audit=audit,dataset_signature=signature,
                    validation=val_metrics,test=test_metrics,timing=timing,parameters=sum(p.numel() for p in model.parameters()),
                    environment=environment,bank_keyspace=bank_keyspace)
    result['total_seconds']=time.perf_counter()-started
    result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    write_json(output/'run_report.json',result)
    print(f'Finished. Outputs: {output}')
    return result
