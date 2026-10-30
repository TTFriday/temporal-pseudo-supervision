"""Versioned, validation-only low-label experiments with audited label access.

Training images remain in original coordinates: torchvision owns resize, padding,
and postprocessing. Historical run_pilot.py results are not reused as evidence.
"""
from __future__ import annotations

import argparse
import collections
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEPS = ROOT / '.deps' / 'site-packages'
if DEPS.exists() and str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))

METHODS = {
    'M1': ('time_coverage', None),
    'M2': ('uncertainty', 'spatial'),
    'M3': ('time_coverage', 'basic'),
    'M4': ('event_aware', 'reliable'),
    'M5': ('time_coverage', 'reliable'),
    'M6': ('event_aware', 'basic'),
}


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_local(path):
    # Only locally generated checkpoints/caches in this experiment directory.
    return torch.load(path, map_location='cpu', weights_only=False)


def frame_key(frame):
    return (frame['plant_id'], frame['replicate_id'], frame['filename'])


def timestamp(frame):
    value = datetime.fromisoformat(frame['timestamp'].replace('Z','+00:00'))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def release_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def image_tensor(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert('RGB')).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.)


class LabelStore:
    """The sole mask reader; algorithms are passed RGB-only metadata."""
    def __init__(self, manifest, protocol):
        self._paths = {frame_key(f): f['mask_path'] for f in manifest}
        self.train = {tuple(k) for k in protocol['train_keys']}
        self.validation = {tuple(k) for k in protocol['validation_keys']}
        self.prompts = {tuple(k) for k in protocol['diagnostic_anchor_keys']}
        self.diagnostic_targets = {tuple(p['target']) for p in protocol.get('diagnostic_pairs',[])}
        self.queried = set()
        self.events = []

    def query(self, keys, budget):
        keys = set(map(tuple, keys))
        if not keys <= self.train:
            raise PermissionError('Query outside training partition')
        if len(self.queried | keys) > budget:
            raise PermissionError('Human annotation budget exceeded')
        self.queried |= keys

    def read(self, key, role):
        key = tuple(key)
        allowed = {'train': self.queried, 'validation': self.validation,
                   'diagnostic_prompt': self.prompts}
        if role not in allowed or key not in allowed[role]:
            raise PermissionError(f'Forbidden mask access: {role} {key}')
        from evaluation.coco_eval import extract_instances_from_mask
        instances = extract_instances_from_mask(self._paths[key])
        # Remove colour and cross-frame IDs; only instance geometry is returned.
        instances = [{'mask': i['mask'].astype(np.uint8), 'bbox': i['bbox'],
                      'area': int(i['mask'].sum())} for i in instances]
        self.events.append({'key': list(key), 'role': role})
        return instances


def build_protocol(manifest, config):
    """Freeze selections using timestamps only, never hidden phenotype labels."""
    split = json.loads(Path(config['split_path']).read_text(encoding='utf-8'))
    by_plant = collections.defaultdict(list)
    for f in manifest:
        by_plant[frame_key(f)[:2]].append(f)
    for frames in by_plant.values():
        frames.sort(key=lambda f: (timestamp(f), f['filename']))
    train_plants = list(map(tuple, split['train_plants']))
    val_plants = list(map(tuple, split['val_plants']))
    test_plants = set(map(tuple, split['test_plants']))
    if set(train_plants) & set(val_plants) or (set(train_plants) | set(val_plants)) & test_plants:
        raise ValueError('Plant leakage in split')
    historical = Path(config.get('historical_pilot_config', ROOT / 'pilot_results' / 'pilot_config.json'))
    if historical.exists():
        old = json.loads(historical.read_text(encoding='utf-8'))
        if config['n_train'] > len(old['train_plants']) or config['n_val'] > len(old['val_plants']):
            raise ValueError('Formal expansion requires a separately audited confirmation protocol')
        train_plants = sorted(map(tuple, old['train_plants']))[:config['n_train']]
        val_plants = sorted(map(tuple, old['val_plants']))[:config['n_val']]
        if not set(train_plants) <= set(map(tuple, split['train_plants'])):
            raise ValueError('Historical pilot training split mismatch')
        if not set(val_plants) <= set(map(tuple, split['val_plants'])):
            raise ValueError('Historical pilot validation split mismatch')
    else:
        # Synthetic/unit-test fixtures only; live pilot must supply its history.
        rng = random.Random(config['split_seed'])
        train_plants = sorted(rng.sample(sorted(train_plants), config['n_train']))
        val_plants = sorted(rng.sample(sorted(val_plants), config['n_val']))
    train = [f for p in train_plants for f in by_plant[p]]
    seed_keys = []
    for p in train_plants:
        frames = by_plant[p]
        middle = (timestamp(frames[0]) + timestamp(frames[-1])) / 2
        seed_keys.append(frame_key(min(frames, key=lambda f: (abs(timestamp(f)-middle), f['filename']))))
    validation, prompts, pairs = [], [], []
    for p in val_plants:
        frames = by_plant[p]
        start, end = timestamp(frames[0]), timestamp(frames[-1])
        chosen = []
        for pos in [.2, .4, .6, .8]:
            available = [f for f in frames if frame_key(f) not in chosen]
            f = min(available, key=lambda f: (abs(timestamp(f)-(start+pos*(end-start))), f['filename']))
            chosen.append(frame_key(f))
        validation.extend(chosen)
        # Extra prompt masks are charged to V, separated from all scoring masks.
        for target_key in [chosen[1], chosen[3]]:
            target = next(f for f in frames if frame_key(f) == target_key)
            candidates = [f for f in frames if frame_key(f) not in chosen and
                          0 < abs(timestamp(f)-timestamp(target)) <= config['max_hours']*3600]
            if not candidates:
                continue
            anchor = min(candidates, key=lambda f: (abs(abs(timestamp(f)-timestamp(target))-24*3600),
                                                    timestamp(f) > timestamp(target), f['filename']))
            ak = frame_key(anchor)
            prompts.append(ak)
            pairs.append({'anchor': list(ak), 'target': list(target_key),
                          'hours': abs(timestamp(anchor)-timestamp(target))/3600})
    keys = [frame_key(f) for f in train]
    budget = math.ceil(len(keys)*config['budget_ratio'])
    if budget < len(seed_keys):
        raise ValueError('Budget smaller than common seeds')
    data = {'version': 2, 'train_plants': train_plants, 'val_plants': val_plants,
            'train_keys': keys, 'seed_keys': seed_keys, 'budget': budget,
            'validation_keys': sorted(validation), 'diagnostic_anchor_keys': sorted(set(prompts)),
            'diagnostic_pairs': pairs,
            'validation_label_count': len(set(validation) | set(prompts)),
            'validation_scoring_count': len(validation),
            'test_policy': 'No test RGB/masks or test metrics in development runner',
            'confirmation_status': 'Historical test exposed; formal confirmation split not yet certified',
            'mask_policy': 'Exact documented palette, every nonempty instance, no size pruning',
            'input_geometry': 'Original RGB -> torchvision internal resize/pad -> original postprocess',
            'propagation_geometry': 'Stop at first image-size discontinuity on each side; do not rotate or resize raw masks',
            'quality_gate_denominator': 'All fixed validation scoring frames including those without propagation; threshold 0.30 unchanged',
            'paired_diagnostic': 'Preset paired targets reported separately; never substitute this easier denominator for the gate',
            'diagnostic_budget_amendment': 'Up to 2 extra prompt labels per val plant, selected before scores'}
    data['sha256'] = digest(data)
    return data


def instances_target(instances, size, pseudo=False):
    w, h = size
    boxes, masks, weights, valids = [], [], [], []
    for inst in instances:
        mask = np.asarray(inst['mask'], dtype=np.uint8)
        if mask.shape != (h, w):
            raise ValueError('Mask not in original image coordinates')
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1])
        masks.append(mask)
        weights.append(float(inst.get('q_weight', 1.)))
        valids.append(np.asarray(inst.get('valid_mask', np.ones((h, w), dtype=np.uint8)), dtype=np.uint8))
    out = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
           'labels': torch.ones(len(boxes), dtype=torch.int64),
           'masks': torch.from_numpy(np.stack(masks) if masks else np.zeros((0,h,w),np.uint8)),
           'confidence': torch.tensor(weights, dtype=torch.float32),
           'is_pseudo': bool(pseudo)}
    if pseudo:
        out['valid_mask'] = torch.from_numpy(np.stack(valids) if valids else np.zeros((0,h,w),np.uint8))
    out['area'] = out['masks'].flatten(1).sum(1).float()
    out['iscrowd'] = torch.zeros(len(boxes), dtype=torch.int64)
    return out


def make_model(seed, config):
    from training.maskrcnn_model import get_model
    seed_everything(seed)
    model = get_model(num_classes=2, pretrained_backbone=True)
    model.transform.min_size = (config['image_size'],)
    model.transform.max_size = config['image_size']
    model.roi_heads.detections_per_img = 100
    return model.cuda()


@torch.inference_mode()
def predict(model, frame, flip=None, threshold=.05):
    from evaluation.coco_eval import model_predict_to_instances
    model.eval()
    im = image_tensor(frame['rgb_path']).cuda()
    h, w = im.shape[-2:]
    if flip is not None:
        im = im.flip(flip)
    output = model([im])[0]
    if flip is not None:
        output['masks'] = output['masks'].flip(flip)
        b = output['boxes'].clone()
        if flip == -1:
            output['boxes'][:, 0] = w-b[:,2]
            output['boxes'][:, 2] = w-b[:,0]
        elif flip == -2:
            output['boxes'][:, 1] = h-b[:,3]
            output['boxes'][:, 3] = h-b[:,1]
    instances = model_predict_to_instances(output, (w,h), score_threshold=threshold)
    return [{**p,'mask':np.asarray(p['mask'],dtype=bool)} for p in instances]


def match_masks(first, second, threshold):
    from selection.event_selector import match_instances
    return match_instances([{**p,'mask':np.asarray(p['mask'],dtype=bool)} for p in first],
                           [{**p,'mask':np.asarray(p['mask'],dtype=bool)} for p in second],threshold)


def evaluate(model, keys, frames, labels, outdir):
    from evaluation.coco_eval import LeafCOCOEvaluator, LeafMaskEvaluator
    evaluator = LeafCOCOEvaluator()
    rows = []
    for key in sorted(keys):
        frame = frames[key]
        gt = labels.read(key, 'validation')
        prediction = predict(model, frame)
        evaluator.add_sample(frame['rgb_path'], None, prediction, gt)
        # Avoid confidence-dependent overlap overwrites: best-scoring mask wins.
        with Image.open(frame['rgb_path']) as im:
            w,h = im.size
        pm, gm = np.zeros((h,w),np.int32), np.zeros((h,w),np.int32)
        selected = [p for p in prediction if p['score'] >= .5]
        for i, p in reversed(list(enumerate(selected, 1))):
            pm[np.asarray(p['mask'],bool)] = i
        for i, g in enumerate(gt, 1):
            gm[np.asarray(g['mask'],bool)] = i
        union = np.logical_or.reduce([p['mask'].astype(bool) for p in selected]) if selected else np.zeros((h,w),bool)
        rows.append({'key': list(key), 'SBD': float(LeafMaskEvaluator().compute_sbd(pm,gm)), 'SBD_score_threshold': .5,
                     'count_abs_error_at_0.5': abs(len(selected)-len(gt)),
                     'union_area_abs_error_at_0.5': int(abs(int(union.sum())-int((gm>0).sum())))})
    metrics = evaluator.evaluate()
    metrics['SBD'] = float(np.mean([r['SBD'] for r in rows]))
    metrics['count_MAE_at_0.5'] = float(np.mean([r['count_abs_error_at_0.5'] for r in rows]))
    metrics['AP_points'] = 100*metrics['AP']
    json_write(Path(outdir)/'metrics.json',metrics)
    json_write(Path(outdir)/'per_frame.json',rows)
    exported = evaluator.export_results()
    json_write(Path(outdir)/'coco_predictions.json', exported['predictions'])
    json_write(Path(outdir)/'coco_ground_truth.json', exported['ground_truth'])
    json_write(Path(outdir)/'coco_parameters.json', exported['parameters'])
    return metrics


def scaled_pseudo_loss(losses, weight):
    """An ignored pseudo batch has no loss and must not enter backward."""
    if not losses or weight == 0:
        return None
    loss = sum(losses.values()) * weight
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite pseudo loss')
    return loss


def pseudo_fingerprint(pseudo):
    records = []
    for key, instances in sorted(pseudo.items()):
        items = []
        for item in instances:
            mask = np.ascontiguousarray(item['mask'], dtype=np.uint8)
            valid = np.ascontiguousarray(item.get('valid_mask', np.ones_like(mask)), dtype=np.uint8)
            items.append({'shape':mask.shape,'mask':hashlib.sha256(mask.tobytes()).hexdigest(),
                          'valid':hashlib.sha256(valid.tobytes()).hexdigest(),
                          'q':float(item.get('q_weight',1.))})
        records.append({'key':key,'instances':items})
    return digest(records)


def train_fixed_steps(directory, seed, human_keys, frames, labels, config, pseudo=None):
    from training.partial_supervision import partial_maskrcnn_loss
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    human_keys = sorted(human_keys)
    pseudo = {k:[i for i in v if i.get('q_weight',1)>0 and np.any(i['mask'])]
              for k,v in (pseudo or {}).items()}
    pseudo = {k:v for k,v in pseudo.items() if v}
    pseudo_keys = sorted(pseudo)
    signature = digest({'seed':seed,'human':human_keys,'pseudo':pseudo_fingerprint(pseudo),'config':config})
    done = directory/'complete.json'
    if done.exists():
        meta = json.loads(done.read_text(encoding='utf-8'))
        if meta['signature'] != signature:
            raise RuntimeError('Refusing stale completed checkpoint')
        return meta
    model = make_model(seed, config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=config['amp'])
    start_step, best_ap, best_step = 0, -1., 0
    elapsed_before, human_seen, pseudo_seen = 0., 0, 0
    checkpoint_path = directory/'latest.pt'
    if checkpoint_path.exists():
        ckpt = load_local(checkpoint_path)
        if ckpt['signature'] != signature:
            raise RuntimeError('Resume config/query mismatch')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v): state[k] = v.cuda()
        scaler.load_state_dict(ckpt['scaler'])
        start_step, best_ap, best_step = ckpt['step'],ckpt['best_ap'],ckpt['best_step']
        human_seen, pseudo_seen = ckpt['human_seen'],ckpt['pseudo_seen']
        elapsed_before = ckpt['elapsed_seconds']
        torch.set_rng_state(ckpt['torch_rng'])
        torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        random.setstate(ckpt['python_rng'])
        np.random.set_state(ckpt['numpy_rng'])
        del ckpt
    # Targets are read only through the annotation ledger; all selected count B.
    human_targets = {}
    for k in human_keys:
        with Image.open(frames[k]['rgb_path']) as im:
            instances = labels.read(k,'train')
            random.Random(int(digest(k)[:8],16)).shuffle(instances)
            human_targets[k] = instances_target(instances,im.size)
    pseudo_targets = {}
    for k in pseudo_keys:
        with Image.open(frames[k]['rgb_path']) as im:
            pseudo_targets[k] = instances_target(pseudo[k], im.size, True)
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    last_metrics = None
    amp_skipped_steps = 0
    stale_evals = 0
    patience = config.get('patience', 0)
    for step in range(start_step+1, config['updates']+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = config['lr']*min(1., .1+.9*step/max(config['warmup_steps'],1))
        for group in optimizer.param_groups: group['lr'] = lr
        rng = random.Random(seed*1000003+step)
        hk = rng.choices(human_keys,k=config['human_batch'])
        hi = [image_tensor(frames[k]['rgb_path']).cuda() for k in hk]
        ht = [{name:(value.cuda() if torch.is_tensor(value) else value) for name,value in human_targets[k].items()} for k in hk]
        with torch.autocast('cuda',dtype=torch.float16,enabled=config['amp']):
            human_losses = model(hi,ht)
            human_loss = sum(human_losses.values())
        if not torch.isfinite(human_loss): raise FloatingPointError('Nonfinite human loss')
        scaler.scale(human_loss).backward()
        human_seen += len(hk)
        pseudo_value = 0.
        if pseudo_keys:
            prng = random.Random(seed*1000003+step+7919)
            pk = prng.choices(pseudo_keys,k=config['pseudo_batch'])
            pi = [image_tensor(frames[k]['rgb_path']).cuda() for k in pk]
            pt = [{name:(value.cuda() if torch.is_tensor(value) else value) for name,value in pseudo_targets[k].items()} for k in pk]
            with torch.autocast('cuda',dtype=torch.float16,enabled=config['amp']):
                losses = partial_maskrcnn_loss(model, pi, pt)
                pseudo_loss = scaled_pseudo_loss(losses, config['lambda_pseudo'])
            if pseudo_loss is not None:
                scaler.scale(pseudo_loss).backward()
                pseudo_value = float(pseudo_loss.detach())
                pseudo_seen += len(pk)
            del pi,pt,losses,pseudo_loss
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            amp_skipped_steps += 1
            if amp_skipped_steps > 10:
                raise FloatingPointError(f'AMP skipped {amp_skipped_steps} updates; training unstable')
            print(f'{directory.name} step={step} AMP skipped update ({amp_skipped_steps}/10)',flush=True)
        if step % config['log_every'] == 0 or step == 1:
            print(f'{directory.name} step={step}/{config["updates"]} human={float(human_loss.detach()):.4f} pseudo={pseudo_value:.4f}',flush=True)
        del hi,ht,human_losses,human_loss
        if step % config['eval_every'] == 0 or step == config['updates']:
            last_metrics = evaluate(model,labels.validation,frames,labels,directory/f'validation_{step:06d}')
            if last_metrics['AP'] > best_ap:
                best_ap,best_step = last_metrics['AP'],step
                stale_evals = 0
                save_torch(directory/'best.pt', {'model_state_dict':model.state_dict(),'step':step,
                                                'validation':last_metrics,'signature':signature})
            else:
                stale_evals += 1
            elapsed = elapsed_before+time.monotonic()-began
            save_torch(checkpoint_path, {'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                       'scaler':scaler.state_dict(),'step':step,'best_ap':best_ap,'best_step':best_step,
                       'signature':signature,'human_seen':human_seen,'pseudo_seen':pseudo_seen,
                       'elapsed_seconds':elapsed,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                       'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
            json_write(directory/'progress.json',{'step':step,'best_ap_points':100*best_ap,'best_step':best_step,
                                                'human_seen':human_seen,'pseudo_seen':pseudo_seen,'elapsed_seconds':elapsed})
            if patience and stale_evals >= patience:
                print(f'{directory.name} early stop at step={step} (no improvement for {patience} evals)',flush=True)
                break
    meta = {'signature':signature,'seed':seed,'updates':config['updates'],'human_seen':human_seen,'pseudo_seen':pseudo_seen,
            'n_human_frames':len(human_keys),'n_pseudo_frames':len(pseudo_keys),
            'best_validation_AP':best_ap,'best_step':best_step,
            'cuda_run_wall_hours':(elapsed_before+time.monotonic()-began)/3600,
            'peak_cuda_allocated_GB':torch.cuda.max_memory_allocated()/1e9,
            'checkpoint':str(directory/'best.pt'),'checkpoint_sha256':file_hash(directory/'best.pt')}
    json_write(done,meta)
    del model,optimizer,scaler,human_targets,pseudo_targets
    release_gpu()
    return meta


def load_model(meta, seed, config):
    model = make_model(seed,config)
    state = load_local(meta['checkpoint'])
    model.load_state_dict(state['model_state_dict'])
    del state
    model.eval()
    return model


def geometry_window(sequence, anchor_key, anchor_shape, max_hours):
    """Do not guess geometry across a camera/image-size discontinuity."""
    center = next(i for i,f in enumerate(sequence) if f['filename']==anchor_key)
    keep = {center}
    excluded = []
    for direction in (-1,1):
        index = center + direction
        while 0 <= index < len(sequence):
            frame = sequence[index]
            if abs(timestamp(frame)-timestamp(sequence[center])) > max_hours*3600:
                break
            with Image.open(frame['rgb_path']) as image:
                size = (image.height,image.width)
            if size != tuple(anchor_shape):
                excluded.append(frame['filename'])
                break
            keep.add(index)
            index += direction
    return [sequence[i] for i in sorted(keep)], excluded


def propagate_cache(directory, anchor_keys, frames, labels, config, role='train'):
    """One SAM2 state per human anchor. Never prompts with scoring-target masks."""
    from propagation.sam2_adapter import SAM2Propagator
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    by_plant = collections.defaultdict(list)
    for k,f in frames.items():
        by_plant[k[:2]].append(f)
    result = collections.defaultdict(dict)
    propagator = None
    for ak in sorted(anchor_keys):
        cache = directory/(digest(ak)+'.pt')
        if cache.exists():
            outputs = load_local(cache)
        else:
            if propagator is None:
                propagator = SAM2Propagator(config['sam2_checkpoint'],config['sam2_config'],device='cuda',cache_root=str(directory/'video'))
            instances = labels.read(ak,role)
            # Local IDs do not encode biological identity or colour order.
            random.Random(int(digest(ak)[:8],16)).shuffle(instances)
            if not instances:
                outputs = {}
            else:
                masks = np.stack([i['mask'] for i in instances])
                sequence = [{name:f[name] for name in ('filename','rgb_path','timestamp')}
                            for f in sorted(by_plant[ak[:2]],key=timestamp)]
                sequence, boundaries = geometry_window(sequence,ak[2],masks.shape[-2:],config['max_hours'])
                json_write(directory/(digest(ak)+'.geometry.json'),
                           {'anchor':ak,'stopped_at_geometry_change':boundaries,
                            'included_rgb_frames':[f['filename'] for f in sequence]})
                outputs = propagator.propagate(sequence,ak[2],masks,max_time_hours=config['max_hours'])
            save_torch(cache,outputs)
            print(f'SAM2 anchor {ak[0]}/{ak[1]}/{ak[2]} targets={len(outputs)}',flush=True)
        for filename, instances in outputs.items():
            key = (*ak[:2],filename)
            if key == ak or key not in frames:
                continue
            if abs(timestamp(frames[key])-timestamp(frames[ak])) > config['max_hours']*3600:
                raise AssertionError('Propagation outside time gate')
            result[key][ak] = [{**i,'mask':np.asarray(i['mask'],dtype=bool)}
                               for i in instances if np.any(i['mask'])]
    del propagator
    release_gpu()
    return dict(result)


def nearest_sources(key, raw, frames):
    """Use closest anchor per temporal side; never merge IDs from two anchors."""
    groups = raw.get(key,{})
    if not groups: return []
    time_target = timestamp(frames[key])
    sides = []
    for before in (True,False):
        candidates = [ak for ak in groups if (timestamp(frames[ak]) < time_target) == before]
        if candidates:
            ak = min(candidates,key=lambda k:(abs(timestamp(frames[k])-time_target),k))
            sides.append((ak,groups[ak]))
    return sorted(sides,key=lambda pair:(abs(timestamp(frames[pair[0]])-time_target),pair[0]))


def pseudo_from_sources(predictions, primary, secondary, mode, config):
    accepted, reasons = [],collections.Counter()
    if mode == 'spatial':
        for p in predictions:
            if p['score'] >= config['confidence'] and np.any(p['mask']):
                accepted.append({**p,'q_weight':1.})
            else: reasons['teacher_confidence_or_empty'] += 1
        return accepted,dict(reasons)
    if primary is None:
        return [],{'no_propagation':1}
    # The common pixel-ignore rule is identical for basic/reliable ablations.
    matched = {i:(j,iou) for i,j,iou in match_masks(primary,predictions,0.)}
    dual = {} if secondary is None else {i:(j,iou) for i,j,iou in match_masks(primary,secondary,0.)}
    eligible = [i for i,p in enumerate(primary)
                if p['score'] >= config['propagation_confidence'] and np.any(p['mask'])]
    teacher_ids = [j for j,p in enumerate(predictions) if p['score'] >= config['confidence']]
    reliable = {eligible[i]:(teacher_ids[j],iou) for i,j,iou in match_masks(
        [primary[i] for i in eligible], [predictions[j] for j in teacher_ids],config['iou'])}
    secondary_ids = [] if secondary is None else [j for j,p in enumerate(secondary)
                      if p['score'] >= config['propagation_confidence']]
    reliable_dual = {} if secondary is None else {eligible[i]:(secondary_ids[j],iou)
        for i,j,iou in match_masks([primary[i] for i in eligible],
                                  [secondary[j] for j in secondary_ids],config['iou'])}
    for i,p in enumerate(primary):
        if i not in eligible:
            reasons['propagation_confidence_or_empty'] += 1
            continue
        item = {**p,'q_weight':1.}
        if mode == 'reliable':
            if i not in reliable:
                reasons['teacher_disagreement'] += 1
                continue
            j,iou = reliable[i]
            if secondary is not None and i not in reliable_dual:
                reasons['independent_anchor_disagreement'] += 1
                continue
            item['q_weight'] = float(iou*predictions[j]['score'])
        # Common ignore rule across temporal baselines: uncertain pixels do not
        # become confident foreground/background targets inside an accepted ROI.
        valid = np.ones_like(p['mask'],dtype=np.uint8)
        if i in matched:
            j,_ = matched[i]
            valid[np.asarray(p['mask'],bool) ^ np.asarray(predictions[j]['mask'],bool)] = 0
        if i in dual:
            j,_ = dual[i]
            valid[np.asarray(p['mask'],bool) ^ np.asarray(secondary[j]['mask'],bool)] = 0
        item['valid_mask'] = valid
        item['independent_anchor_evidence'] = secondary is not None
        accepted.append(item)
    return accepted,dict(reasons)


def build_pseudo(directory, keys, teacher, frames, raw, mode, config):
    directory = Path(directory)
    cache = directory/'instances.pt'
    if cache.exists():
        package = load_local(cache)
        json_write(directory/'statistics.json',package['statistics'])
        return package['instances'],package['statistics']
    result, reasons = {},collections.Counter()
    count_single,count_dual = 0,0
    for key in sorted(keys):
        sources = nearest_sources(key,raw,frames) if mode != 'spatial' else []
        if mode != 'spatial' and not sources:
            reasons['no_anchor_in_window'] += 1
            continue
        preds = predict(teacher,frames[key],threshold=.05)
        primary = sources[0][1] if sources else None
        secondary = sources[1][1] if len(sources)>1 else None
        accepted, rejected = pseudo_from_sources(preds,primary,secondary,mode,config)
        reasons.update(rejected)
        if accepted:
            result[key] = accepted
            if secondary is None: count_single += len(accepted)
            else: count_dual += len(accepted)
    stats = {'mode':mode,'n_candidate_frames':len(keys),'n_frames':len(result),
             'n_instances':sum(map(len,result.values())), 'single_anchor_or_spatial_instances':count_single,
             'independent_dual_anchor_instances':count_dual,'rejections':dict(reasons)}
    save_torch(cache,{'instances':result,'statistics':stats})
    json_write(directory/'statistics.json',stats)
    return result,stats


def diagnostic_quality(pseudo, labels):
    n_gt,n_accepted,n_good,n_small,n_small_good = 0,0,0,0,0
    rows = []
    for key in sorted(labels.validation):
        gt = labels.read(key,'validation')
        accepted = pseudo.get(key,[])
        pairs = match_masks(accepted,gt,.75)
        good_gt = {j for _,j,_ in pairs}
        small = {j for j,g in enumerate(gt) if g['area'] < 32**2}
        n_gt += len(gt); n_accepted += len(accepted); n_good += len(pairs)
        n_small += len(small); n_small_good += len(small & good_gt)
        rows.append({'key':list(key),'n_gt':len(gt),'n_accepted':len(accepted),'n_good':len(pairs)})
    paired = [r for r in rows if tuple(r['key']) in labels.diagnostic_targets]
    paired_gt = sum(r['n_gt'] for r in paired)
    paired_accepted = sum(r['n_accepted'] for r in paired)
    paired_good = sum(r['n_good'] for r in paired)
    return {'n_gt':n_gt,'n_accepted':n_accepted,'n_good':n_good,
            'denominator_policy':'All frozen validation scoring frames, including unavailable propagation',
            'precision_iou_075':n_good/n_accepted if n_accepted else 0.,
            'coverage_iou_075':n_good/n_gt if n_gt else 0.,
            'paired_target_diagnostic':{'n_frames':len(paired),'n_gt':paired_gt,'n_accepted':paired_accepted,
                'n_good':paired_good,'precision':paired_good/paired_accepted if paired_accepted else 0.,
                'coverage':paired_good/paired_gt if paired_gt else 0.,'used_for_gate':False},
            'small_coverage_iou_075':n_small_good/n_small if n_small else None,
            'n_small_gt':n_small,'per_frame':rows}


def score_candidates(directory, teacher, train_frames, selected, raw, config):
    from selection.event_selector import compute_signals
    cache = Path(directory)/'signals.pt'
    if cache.exists(): return load_local(cache)
    results = {}
    for key,frame in sorted(train_frames.items()):
        if key in selected: continue
        original = predict(teacher,frame,threshold=.5)
        augmented = [predict(teacher,frame,flip=axis,threshold=.5) for axis in (-1,-2)]
        sources = nearest_sources(key,raw,train_frames)
        prop = sources[0][1] if sources else None
        results[key] = compute_signals(original,augmented,prop,conf_threshold=config['confidence'])
    save_torch(cache,results)
    return results


def gate_summary(runs, seeds):
    rows = {(r['method'],r['seed']):r for r in runs}
    required = {(m,s) for m in METHODS for s in seeds}
    if not required <= rows.keys():
        return {'status':'incomplete','formal_allowed':False,'completed':len(rows),'required':len(required)}
    means = {m:float(np.mean([rows[m,s]['student']['best_validation_AP'] for s in seeds])) for m in METHODS}
    baseline = max(('M2','M3'),key=lambda m:means[m])
    diffs = [100*(rows['M4',s]['student']['best_validation_AP']-rows[baseline,s]['student']['best_validation_AP']) for s in seeds]
    performance = np.mean(diffs)>=1. and all(d>0 for d in diffs)
    qualities = [rows['M4',s]['diagnostic'] for s in seeds]
    quality = all(q['precision_iou_075']>=.85 and q['coverage_iou_075']>=.30 for q in qualities)
    return {'status':'pass' if performance and quality else 'stop_for_review','formal_allowed':bool(performance and quality),
            'reference_only_not_significance':True,'baseline':baseline,'AP_mean_points':{m:100*v for m,v in means.items()},
            'paired_AP_point_differences':diffs,'performance_gate':bool(performance),'quality_gate':bool(quality),
            'confirmation_split_still_required':True}


def audited_manifest(manifest, config):
    """Validate the audit, then pass only its key whitelist to algorithms."""
    from data_audit.check_coco_consistency import read_algorithm_facing_result
    path = config.get('audit_path')
    if not path or file_hash(path) != config.get('audit_sha256'):
        raise RuntimeError('A matching completed data audit is required')
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    provenance = report['provenance']
    if (not provenance['inputs_unchanged'] or report['summary']['global_problem_count']
            or provenance['manifest']['sha256'] != file_hash(config['manifest_path'])
            or provenance['audit_code']['sha256'] != file_hash(provenance['audit_code']['path'])):
        raise RuntimeError('Audit source mismatch or unresolved global errors')
    # This GT-bearing report is never passed to the selector or model.
    del report
    safe = read_algorithm_facing_result(path)
    valid = {tuple(k.split('/')) for k in safe['valid_frame_keys']}
    excluded = {tuple(k.split('/')) for k in safe['excluded_frame_keys']}
    actual = {frame_key(f) for f in manifest}
    if valid & excluded or (valid | excluded) != actual or len(actual) != len(manifest):
        raise RuntimeError('Incomplete or duplicate audit frame universe')
    return [f for f in manifest if frame_key(f) in valid]


def sam2_fingerprint(config):
    path = ROOT/'propagation'/'sam2_provenance.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    expected = record['checkpoint']
    if (file_hash(config['sam2_checkpoint']) != expected['sha256']
            or config['sam2_config'] != expected['model_config']):
        raise RuntimeError('SAM2 checkpoint/config differs from verified adapter')
    for name, expected_hash in record['source']['installed_source_hashes'].items():
        if file_hash(DEPS/name) != expected_hash:
            raise RuntimeError(f'SAM2 source changed: {name}')
    return {'provenance_sha256':file_hash(path),'checkpoint_sha256':expected['sha256'],
            'config_sha256':file_hash(expected['config_path'])}


def run_experiment(config, output, mode, selected_methods, seeds):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required: refusing CPU training')
    output = Path(output).resolve()
    if output == (ROOT/'pilot_results').resolve() or (ROOT/'archives').resolve() in output.parents:
        raise ValueError('Historical output is protected')
    output.mkdir(parents=True,exist_ok=True)
    lock = output/'running.lock'
    # Exclusive live-process lock, never automatically break a possibly live lock.
    fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try:
        manifest = json.loads(Path(config['manifest_path']).read_text(encoding='utf-8'))
        if mode != 'prepare':
            manifest = audited_manifest(manifest,config)
        protocol = build_protocol(manifest,config)
        sources = {str(p.relative_to(ROOT)):file_hash(p) for sub in ('training','selection','evaluation','propagation')
                   for p in (ROOT/sub).glob('*.py')}
        sources['run_pilot.py'] = file_hash(ROOT/'run_pilot.py')
        temporal = any(METHODS[m][1] in ('basic','reliable') for m in selected_methods)
        sam2 = sam2_fingerprint(config) if temporal and mode != 'prepare' else None
        provenance = {'config':config,'protocol_sha256':protocol['sha256'],'source_sha256':sources,'sam2':sam2,
                      'manifest_sha256':file_hash(config['manifest_path']),'split_sha256':file_hash(config['split_path']),
                      'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,
                      'python':sys.version,'gpu':torch.cuda.get_device_name(0),
                      'cuda':torch.version.cuda,'seeds':seeds,'mode':mode}
        record = output/'provenance.json'
        if record.exists() and json.loads(record.read_text(encoding='utf-8')) != provenance:
            raise RuntimeError('Output provenance differs; use a new directory')
        json_write(record,provenance)
        json_write(output/'protocol.json',protocol)
        # Do not forward GT-derived manifest statistics to models or selectors.
        permitted = set(map(tuple,protocol['train_keys'])) | set(map(tuple,protocol['validation_keys'])) | set(map(tuple,protocol['diagnostic_anchor_keys']))
        val_plants = set(map(tuple,protocol['val_plants']))
        frames = {frame_key(f):{k:f[k] for k in ('plant_id','replicate_id','filename','timestamp','rgb_path')}
                  for f in manifest if frame_key(f) in permitted or frame_key(f)[:2] in val_plants}
        train_frames = {k:frames[k] for k in map(tuple,protocol['train_keys'])}
        seed_keys = set(map(tuple,protocol['seed_keys']))
        if mode == 'prepare':
            return {'status':'protocol_frozen','budget':protocol['budget'],'V':protocol['validation_label_count']}
        # Audit is an explicit precondition for a real pilot; smoke is synthetic/
        # infrastructure evidence, not a substitute for this gate.
        if mode == 'pilot' and not config.get('audit_cleared',False):
            raise RuntimeError('Data audit must be cleared before scientific pilot')
        all_runs = []
        result_path = output/'results_intermediate.json'
        if result_path.exists(): all_runs = json.loads(result_path.read_text(encoding='utf-8'))
        from selection.event_selector import select_query_keys
        for seed in seeds:
            seed_dir = output/f'seed_{seed}'
            store = LabelStore(manifest,protocol)
            store.query(seed_keys,protocol['budget'])
            seed_teacher = train_fixed_steps(seed_dir/'seed_teacher',seed,seed_keys,frames,store,config)
            seed_raw = None
            for selector in dict.fromkeys(METHODS[m][0] for m in selected_methods):
                method_ids = [m for m in selected_methods if METHODS[m][0]==selector]
                directory = seed_dir/selector
                query_path = directory/'query.json'
                if query_path.exists():
                    query = json.loads(query_path.read_text(encoding='utf-8'))
                    selected = set(map(tuple,query['selected_keys']))
                else:
                    if selector in ('uncertainty','event_aware'):
                        if selector == 'event_aware' and seed_raw is None:
                            seed_raw = propagate_cache(seed_dir/'seed_sam2',seed_keys,train_frames,store,config)
                        model = load_model(seed_teacher,seed,config)
                        scoring_raw = seed_raw if selector == 'event_aware' else {}
                        signals = score_candidates(seed_dir/f'seed_scores_{selector}',model,train_frames,seed_keys,scoring_raw,config)
                        del model; release_gpu()
                    else: signals = None
                    new_keys,audit = select_query_keys(list(train_frames.values()),seed_keys,protocol['budget'],selector,signals=signals,eta=config['eta'])
                    selected = seed_keys | set(map(tuple,new_keys))
                    query = {'selected_keys':sorted(selected),'seed_keys':sorted(seed_keys),'audit':audit,
                             'selector':selector,'teacher_source_sha256':seed_teacher['checkpoint_sha256']}
                    json_write(query_path,query)
                if len(selected)!=protocol['budget']:
                    raise AssertionError('Incorrect label budget')
                # Each selector has its own budget ledger; no labels spill over.
                store = LabelStore(manifest,protocol)
                store.query(selected,protocol['budget'])
                teacher = train_fixed_steps(directory/'teacher',seed,selected,frames,store,config)
                need_temporal = any(METHODS[m][1] in ('basic','reliable') for m in method_ids)
                raw = propagate_cache(directory/'sam2',selected,train_frames,store,config) if need_temporal else {}
                val_frames = {k:f for k,f in frames.items() if k[:2] in val_plants}
                val_raw = propagate_cache(output/'validation_sam2',store.prompts,val_frames,store,config,'diagnostic_prompt') if need_temporal else {}
                for method in method_ids:
                    if any(r['method']==method and r['seed']==seed for r in all_runs): continue
                    mode_pseudo = METHODS[method][1]
                    mdir = directory/method
                    if mode_pseudo:
                        model = load_model(teacher,seed,config)
                        pseudo,stats = build_pseudo(mdir/'pseudo',set(train_frames)-selected,model,train_frames,raw,mode_pseudo,config)
                        if mode_pseudo in ('basic','reliable'):
                            val_pseudo,_ = build_pseudo(mdir/'diagnostic',store.validation,model,val_frames,val_raw,mode_pseudo,config)
                            quality = diagnostic_quality(val_pseudo,store)
                        else: quality = None
                        del model; release_gpu()
                    else:
                        pseudo,stats,quality = {},{'n_frames':0,'n_instances':0},None
                    student = train_fixed_steps(mdir/'student',seed,selected,frames,store,config,pseudo)
                    result = {'method':method,'seed':seed,'selector':selector,'teacher':teacher,'student':student,
                              'B':len(selected),'V':protocol['validation_label_count'],'query_sha256':digest(sorted(selected)),
                              'pseudo':stats,'diagnostic':quality,'validation_only':True}
                    json_write(mdir/'result.json',result)
                    json_write(mdir/'label_access.json',store.events)
                    all_runs.append(result)
                    json_write(result_path,all_runs)
                    json_write(output/'gate.json',gate_summary(all_runs,seeds))
                    del pseudo; release_gpu()
                del raw,val_raw; release_gpu()
        json_write(output/'results_final.json',{'runs':all_runs,'gate':gate_summary(all_runs,seeds),'protocol_sha256':protocol['sha256']})
        return gate_summary(all_runs,seeds)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Audited validation-only GPU pilot')
    parser.add_argument('--config',default=str(ROOT/'configs'/'validated_pilot.json'))
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--mode',choices=['prepare','smoke','pilot'],default='prepare')
    parser.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS))
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,42])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    print(json.dumps(run_experiment(config,args.output_dir,args.mode,args.methods,args.seeds),indent=2),flush=True)


if __name__ == '__main__':
    main()
