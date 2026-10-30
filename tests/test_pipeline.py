"""Protocol/ledger integration regressions; no CPU model training."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from training.pipeline import (LabelStore, build_protocol, frame_key, gate_summary,
                               instances_target, json_write, pseudo_from_sources)


def fixture_protocol(tmp_path):
    plants = [('g','r1'),('g','r2'),('g','r3')]
    split = {'train_plants':[plants[0]],'val_plants':[plants[1]],'test_plants':[plants[2]]}
    path = tmp_path/'split.json'
    json_write(path,split)
    manifest = []
    for p in plants:
        for j in range(20):
            manifest.append({'plant_id':p[0],'replicate_id':p[1],'filename':f'{j:03d}.png',
                             'timestamp':(datetime(2022,1,1)+timedelta(hours=12*j)).isoformat(),
                             'rgb_path':f'{p}/{j}.png','mask_path':f'{p}/{j}_mask.png',
                             'n_leaves':999})
    cfg = {'split_path':str(path),'historical_pilot_config':str(tmp_path/'absent.json'),
           'n_train':1,'n_val':1,'split_seed':42,'budget_ratio':.1,'max_hours':48}
    return manifest,build_protocol(manifest,cfg),cfg


def test_protocol_budget_and_separated_validation_prompts(tmp_path):
    manifest,p,cfg = fixture_protocol(tmp_path)
    assert p['budget']==2
    assert len(p['seed_keys'])==1
    val = set(map(tuple,p['validation_keys']))
    prompts = set(map(tuple,p['diagnostic_anchor_keys']))
    assert len(val)==4 and 1<=len(prompts)<=2
    assert val.isdisjoint(prompts)
    assert p['validation_label_count']==len(val|prompts)
    assert all(pair['hours']<=48 for pair in p['diagnostic_pairs'])
    assert all(k[1]!='r3' for k in p['train_keys']+p['validation_keys'])
    _,p2,_ = fixture_protocol(tmp_path)
    assert p['sha256']==p2['sha256']


def test_ledger_forbids_unqueried_and_test_and_target_prompt(tmp_path):
    manifest,p,cfg = fixture_protocol(tmp_path)
    labels = LabelStore(manifest,p)
    key = tuple(p['train_keys'][0])
    with pytest.raises(PermissionError): labels.read(key,'train')
    with pytest.raises(PermissionError): labels.query([('g','r3','000.png')],2)
    with pytest.raises(PermissionError): labels.query(p['train_keys'][:3],2)
    with pytest.raises(PermissionError): labels.read(p['validation_keys'][0],'diagnostic_prompt')
    labels.query([key],2)
    assert labels.queried=={key}


def test_target_keeps_single_pixel_instance_and_exclusive_bbox():
    mask = np.zeros((17,31),np.uint8)
    mask[16,30]=1
    target = instances_target([{'mask':mask,'q_weight':.25}],(31,17),True)
    assert target['boxes'].tolist()==[[30.,16.,31.,17.]]
    assert target['area'].tolist()==[1.]
    assert target['confidence'].tolist()==[.25]
    assert target['valid_mask'].shape==(1,17,31)
    with pytest.raises(ValueError): instances_target([{'mask':mask}],(17,31))


def test_empty_target_retains_original_geometry():
    target = instances_target([],(31,17),True)
    assert target['masks'].shape==(0,17,31)
    assert target['boxes'].shape==(0,4)


def test_gate_incomplete_not_passing():
    assert not gate_summary([], [17,42])['formal_allowed']


def test_gate_uses_stronger_baseline_and_paired_seeds():
    runs=[]
    for seed in (17,42):
        for method in ('M1','M2','M3','M4','M5','M6'):
            ap={'M1':.1,'M2':.2,'M3':.25,'M4':.27,'M5':.2,'M6':.2}[method]
            runs.append({'method':method,'seed':seed,'student':{'best_validation_AP':ap},
                         'diagnostic':{'precision_iou_075':.9,'coverage_iou_075':.4}})
    result=gate_summary(runs,[17,42])
    assert result['baseline']=='M3' and result['formal_allowed']
    runs[-3]['student']['best_validation_AP']=.24
    assert not gate_summary(runs,[17,42])['formal_allowed']


def test_spatial_pseudo_actual_unfiltered_count():
    mask=np.zeros((12,15),np.uint8); mask[2:8,2:8]=1
    accepted,reasons=pseudo_from_sources([{'mask':mask,'score':.9},{'mask':mask,'score':.2}],
                                        None,None,'spatial',{'confidence':.8})
    assert len(accepted)==1 and reasons['teacher_confidence_or_empty']==1
    assert accepted[0]['q_weight']==1.


def test_independent_anchor_contradiction_rejected():
    cfg={'confidence':.8,'propagation_confidence':.8,'iou':.7}
    mask=np.zeros((12,15),np.uint8); mask[2:7,2:7]=1
    other=np.zeros_like(mask); other[7:11,8:13]=1
    p={'mask':mask,'bbox':[2,2,5,5],'score':.95}
    q={'mask':other,'bbox':[8,7,5,4],'score':.95}
    accepted,reasons=pseudo_from_sources([p],[p],[q],'reliable',cfg)
    assert not accepted and reasons['independent_anchor_disagreement']==1
    accepted,_=pseudo_from_sources([p],[p],None,'reliable',cfg)
    assert len(accepted)==1 and accepted[0]['q_weight']==pytest.approx(.95)
    assert not accepted[0]['independent_anchor_evidence']


def test_empty_pseudo_loss_skips_backward():
    import torch
    from training.pipeline import scaled_pseudo_loss
    assert scaled_pseudo_loss({}, .5) is None
    assert scaled_pseudo_loss({'mask':torch.tensor(1.)}, 0) is None
    value = torch.tensor(2., requires_grad=True)
    loss = scaled_pseudo_loss({'mask':value}, .5)
    loss.backward()
    assert value.grad.item() == .5


def test_pseudo_fingerprint_pins_pixels_weights_and_validity():
    from training.pipeline import pseudo_fingerprint
    mask = np.zeros((5,7),np.uint8); mask[2,3] = 1
    item = {'mask':mask,'q_weight':.8}
    pseudo = {('g','r','a.png'):[item]}
    original = pseudo_fingerprint(pseudo)
    item['q_weight'] = .7
    assert pseudo_fingerprint(pseudo) != original
    item['q_weight'] = .8
    item['valid_mask'] = np.zeros_like(mask)
    assert pseudo_fingerprint(pseudo) != original
    del item['valid_mask']
    mask[1,3] = 1
    assert pseudo_fingerprint(pseudo) != original


def test_propagation_strips_metadata_and_preserves_mask_coordinates(tmp_path, monkeypatch):
    from training.pipeline import propagate_cache
    import propagation.sam2_adapter as adapter
    key = ('g','r','a.png'); target = ('g','r','b.png')
    frames = {key:dict(plant_id='g',replicate_id='r',filename='a.png',rgb_path='a.png',timestamp='2022-01-01T00:00:00'),
              target:dict(plant_id='g',replicate_id='r',filename='b.png',rgb_path='b.png',timestamp='2022-01-01T06:00:00')}
    from PIL import Image
    for frame in frames.values():
        path = tmp_path/frame['filename']
        Image.new('RGB',(9,8)).save(path)
        frame['rgb_path'] = str(path)
    class Labels:
        def read(self,k,role):
            assert k == key and role == 'train'
            return [{'mask':np.ones((8,9),np.uint8)}]
    class Fake:
        def __init__(self,*args,**kwargs): pass
        def propagate(self,sequence,anchor,masks,max_time_hours):
            assert all(set(f)=={'filename','rgb_path','timestamp'} for f in sequence)
            assert anchor=='a.png' and masks.shape==(1,8,9)
            return {'b.png':[{'mask':masks[0],'score':.9}]}
    monkeypatch.setattr(adapter,'SAM2Propagator',Fake)
    result = propagate_cache(tmp_path,[key],frames,Labels(),{'sam2_checkpoint':'fake','sam2_config':'fake','max_hours':48})
    assert result[target][key][0]['mask'].dtype==np.bool_
